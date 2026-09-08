"""
Tier 3 -- the reasoning pass.

Runs once per shift, not once per part. It reads the full inspection log and
produces the thing a plant actually needs and never gets: a written account of
what went wrong today and what it suggests about the process.

This is where the economics of an on-device NPU stop being a privacy argument
and start being an arithmetic one. A shift log is 500-2,000 events. Summarising
it costs tokens. Doing that per line, per shift, per day, across a company with
eleven lines is roughly 120,000 LLM calls a year -- a recurring cloud bill that
an MSME will simply decline to pay, which is why this analysis does not exist in
those plants today. On the NPU the marginal cost is electricity, so it runs
every shift on every line and nobody has to justify it.

Two things happen here that Tier 1 cannot do alone:

  * Statistical drift detection, computed in numpy, not by the model. If the
    mean anomaly score of PASSING parts is climbing, the process is moving
    before it starts producing scrap. That is a leading indicator, and it is
    arithmetic -- so we compute it and hand the model the result rather than
    asking a language model to do statistics.

  * Root-cause hypothesis generation, which is genuinely a language task:
    correlating "17 scratches, all lower-left, all after 14:00" with the fact
    that the fixture was changed at 13:45.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_LLM = "ai-hub-models/Qwen3-4B-Instruct-2507"

SYSTEM_PROMPT = """You are a manufacturing quality engineer writing an end-of-shift \
report for a small factory in India. You are given aggregated inspection statistics \
that have already been computed. Do not recompute them and do not invent numbers.

Write four short sections with these exact headings:
## Summary
## Defect Pattern
## Likely Causes
## Recommended Actions

Rules:
- Under 320 words total.
- Every claim must trace to the supplied data. If the data does not support a \
cause, say the data is insufficient rather than guessing.
- Recommended actions must be things a shift supervisor can do tomorrow morning \
with the tools in a small workshop.
- Plain language. The reader is a supervisor, not a statistician."""


@dataclass
class InspectionEvent:
    """One part, one record."""

    timestamp: float
    verdict: str
    score: float
    defect_type: str | None = None
    location: str | None = None
    description: str | None = None

    @property
    def hour(self) -> int:
        return datetime.fromtimestamp(self.timestamp).hour


@dataclass
class ShiftStats:
    total: int
    passed: int
    review: int
    failed: int
    yield_pct: float
    defect_types: dict[str, int]
    locations: dict[str, int]
    hourly_fail_rate: dict[int, float]
    pass_score_trend: dict[str, float]
    drift_detected: bool
    drift_note: str
    peak_hour: int | None
    extras: dict = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        lines = [
            f"Parts inspected: {self.total}",
            f"Passed: {self.passed} | Sent to review: {self.review} | Failed: {self.failed}",
            f"First-pass yield: {self.yield_pct:.1f}%",
        ]
        if self.defect_types:
            top = ", ".join(f"{k} x{v}" for k, v in
                            sorted(self.defect_types.items(), key=lambda kv: -kv[1]))
            lines.append(f"Defect types observed: {top}")
        if self.locations:
            top = ", ".join(f"{k} x{v}" for k, v in
                            sorted(self.locations.items(), key=lambda kv: -kv[1])[:5])
            lines.append(f"Defect locations: {top}")
        if self.hourly_fail_rate:
            hrs = ", ".join(f"{h:02d}:00 -> {r:.0f}%" for h, r in sorted(self.hourly_fail_rate.items()))
            lines.append(f"Fail rate by hour: {hrs}")
        if self.peak_hour is not None:
            lines.append(f"Worst hour: {self.peak_hour:02d}:00")
        lines.append(
            "Anomaly-score drift among PASSING parts: "
            f"first quarter mean {self.pass_score_trend.get('first_quarter', 0):.4f}, "
            f"last quarter mean {self.pass_score_trend.get('last_quarter', 0):.4f} "
            f"({self.pass_score_trend.get('change_pct', 0):+.1f}%)"
        )
        lines.append(f"Drift alarm: {'YES -- ' + self.drift_note if self.drift_detected else 'no'}")
        for k, v in self.extras.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)


def compute_stats(events: list[InspectionEvent], *, drift_threshold_pct: float = 12.0,
                  extras: dict | None = None) -> ShiftStats:
    """Everything numeric happens here, in numpy, before any model sees anything."""
    if not events:
        raise ValueError("No inspection events to summarise.")

    events = sorted(events, key=lambda e: e.timestamp)
    total = len(events)
    passed = sum(1 for e in events if e.verdict == "PASS")
    review = sum(1 for e in events if e.verdict == "REVIEW")
    failed = sum(1 for e in events if e.verdict == "FAIL")

    defect_types = Counter(e.defect_type for e in events if e.defect_type)
    locations = Counter(e.location for e in events if e.location)

    hourly: dict[int, list[int]] = {}
    for e in events:
        hourly.setdefault(e.hour, []).append(1 if e.verdict == "FAIL" else 0)
    hourly_fail_rate = {h: 100.0 * sum(v) / len(v) for h, v in hourly.items()}
    peak_hour = max(hourly_fail_rate, key=hourly_fail_rate.get) if hourly_fail_rate else None

    # Drift is measured on PASSING parts only. Failures are the symptom; the
    # slow upward creep of good parts is the disease, and it shows up first.
    pass_scores = [e.score for e in events if e.verdict == "PASS"]
    trend: dict[str, float] = {}
    drift, note = False, ""
    if len(pass_scores) >= 20:
        q = max(1, len(pass_scores) // 4)
        first = float(np.mean(pass_scores[:q]))
        last = float(np.mean(pass_scores[-q:]))
        change = 100.0 * (last - first) / max(first, 1e-6)
        trend = {"first_quarter": first, "last_quarter": last, "change_pct": change}
        if change > drift_threshold_pct:
            drift = True
            note = (f"mean anomaly score of passing parts rose {change:.1f}% across the "
                    "shift; the process is moving toward the reject threshold even "
                    "though parts are still passing")

    return ShiftStats(
        total=total, passed=passed, review=review, failed=failed,
        yield_pct=100.0 * passed / total,
        defect_types=dict(defect_types), locations=dict(locations),
        hourly_fail_rate=hourly_fail_rate,
        pass_score_trend=trend, drift_detected=drift, drift_note=note,
        peak_hour=peak_hour, extras=extras or {},
    )


class ShiftReporter:
    def __init__(self, model_id: str = DEFAULT_LLM, *, device_map: str = "qairt",
                 max_new_tokens: int = 700):
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._unavailable_reason: str | None = None

    def _ensure(self):
        if self._model is not None or self._unavailable_reason:
            return
        try:
            from geniex import AutoModelForCausalLM
        except ImportError:
            self._unavailable_reason = "geniex not installed"
            return
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=self.device_map)
        except Exception as exc:  # pragma: no cover
            self._unavailable_reason = str(exc)

    def generate(self, stats: ShiftStats, *, context: str = "", language: str = "English") -> str:
        self._ensure()
        if self._model is None:
            return _fallback_report(stats, self._unavailable_reason or "model unavailable")

        user = f"Shift data:\n{stats.to_prompt_block()}"
        if context:
            user += f"\n\nOperational context supplied by the supervisor:\n{context}"
        if language.lower() != "english":
            user += f"\n\nWrite the report in {language}. Keep the four headings in English."

        prompt = self._model.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            add_generation_prompt=True,
        )
        t0 = time.perf_counter()
        out = self._model.generate(prompt, max_new_tokens=self.max_new_tokens)
        log.info("Shift report generated in %.1f s (%.1f tok/s)",
                 time.perf_counter() - t0, getattr(out.profile, "decode_speed", 0.0))
        return out.text.strip()

    def close(self) -> None:
        if self._model is not None:
            self._model.close()
            self._model = None


def _fallback_report(stats: ShiftStats, reason: str) -> str:
    """
    A deterministic report when no LLM is present.

    This is not a stub. A line that stops producing its shift record because a
    model failed to load is worse than one that produces a plainer record, so
    the statistics -- which are the audit trail -- are always emitted.
    """
    top = sorted(stats.defect_types.items(), key=lambda kv: -kv[1])
    lines = [
        "## Summary",
        f"{stats.total} parts inspected. First-pass yield {stats.yield_pct:.1f}% "
        f"({stats.failed} failed, {stats.review} sent to manual review).",
        "",
        "## Defect Pattern",
        ("Most frequent: " + ", ".join(f"{k} ({v})" for k, v in top[:3])) if top
        else "No classified defects recorded.",
    ]
    if stats.peak_hour is not None:
        lines.append(f"Highest fail rate at {stats.peak_hour:02d}:00 "
                     f"({stats.hourly_fail_rate[stats.peak_hour]:.0f}%).")
    lines += ["", "## Likely Causes"]
    lines.append(stats.drift_note.capitalize() + "." if stats.drift_detected
                 else "No statistically significant drift detected this shift.")
    lines += [
        "", "## Recommended Actions",
        "- Review the flagged parts retained in the review bin.",
        "- Re-check fixture seating and lighting at the hour with the highest fail rate.",
        "",
        f"_Narrative generation was unavailable ({reason}); statistics above are complete._",
    ]
    return "\n".join(lines)


def export_events(events: list[InspectionEvent], path) -> None:
    """JSONL, because an inspection log should be greppable without our software."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps({
                "timestamp": e.timestamp, "iso": datetime.fromtimestamp(e.timestamp).isoformat(),
                "verdict": e.verdict, "score": round(e.score, 5),
                "defect_type": e.defect_type, "location": e.location,
                "description": e.description,
            }) + "\n")
