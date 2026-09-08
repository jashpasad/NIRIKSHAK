"""
The cascade.

This file is the argument of the whole project, so it is worth stating plainly.

A naive design runs a vision-language model on every part. On a Snapdragon X2
Elite that is roughly 400 ms per part, which caps the line at about 2.5 parts
per second and pins the NPU at full draw all shift. It also produces a
description of 950 parts that were perfectly fine, which nobody reads.

NIRIKSHAK instead spends compute in proportion to uncertainty:

    Tier 0  Gate       Arduino UNO Q   ~2 ms    every frame, part-present only
    Tier 1  Screen     Hexagon NPU     ~10 ms   every part
    Tier 2  Explain    Hexagon NPU     ~400 ms  only the review/fail band
    Tier 3  Reason     Hexagon NPU     ~25 s    once per shift

At a realistic 95% yield with a 3% review band, Tier 2 fires on roughly 8% of
parts. Mean cost per part is therefore about 10 + 0.08 x 400 = 42 ms, not
400 ms -- an order of magnitude, obtained purely by ordering the models rather
than by making any one of them faster.

The escalation policy is deliberately explicit and inspectable. On an audited
line, "why did this part get escalated" must have an answer that is a rule, not
a model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .tier1_screen.detector import AnomalyDetector, InspectionResult
from .tier2_explain.vlm import DefectExplainer, Explanation
from .tier3_reason.shift_report import InspectionEvent

log = logging.getLogger(__name__)


@dataclass
class CascadePolicy:
    """When to spend the expensive model."""

    explain_on_fail: bool = True
    explain_on_review: bool = True
    # A part that only just crosses into REVIEW is usually lighting or jitter.
    # Requiring a minimum severity stops Tier 2 from being woken by noise.
    min_severity_for_explain: float = 0.15
    # Hard ceiling on escalations per minute. If the line goes badly wrong, we
    # would rather keep screening every part at 10 ms than fall behind trying to
    # narrate a flood of failures. Throughput protection beats completeness.
    max_explanations_per_min: int = 30


@dataclass
class PartRecord:
    part_id: str
    result: InspectionResult
    explanation: Explanation | None = None
    escalated: bool = False
    escalation_reason: str = ""
    total_ms: float = 0.0


@dataclass
class CascadeTelemetry:
    parts: int = 0
    escalations: int = 0
    tier1_ms: list[float] = field(default_factory=list)
    tier2_ms: list[float] = field(default_factory=list)
    throttled: int = 0

    def summary(self) -> dict:
        t1 = np.asarray(self.tier1_ms) if self.tier1_ms else np.zeros(1)
        t2 = np.asarray(self.tier2_ms) if self.tier2_ms else np.zeros(1)
        escalation_rate = self.escalations / self.parts if self.parts else 0.0
        mean_cost = float(t1.mean() + escalation_rate * (t2.mean() if self.tier2_ms else 0.0))
        return {
            "parts": self.parts,
            "escalations": self.escalations,
            "escalation_rate_pct": round(100 * escalation_rate, 2),
            "throttled": self.throttled,
            "tier1_mean_ms": round(float(t1.mean()), 2),
            "tier1_p95_ms": round(float(np.percentile(t1, 95)), 2),
            "tier2_mean_ms": round(float(t2.mean()), 2) if self.tier2_ms else None,
            "effective_ms_per_part": round(mean_cost, 2),
            "sustainable_parts_per_min": round(60_000 / mean_cost, 1) if mean_cost > 0 else None,
        }


class InspectionCascade:
    def __init__(
        self,
        detector: AnomalyDetector,
        explainer: DefectExplainer | None = None,
        policy: CascadePolicy | None = None,
        *,
        part_name: str = "part",
    ):
        self.detector = detector
        self.explainer = explainer
        self.policy = policy or CascadePolicy()
        self.part_name = part_name
        self.telemetry = CascadeTelemetry()
        self.events: list[InspectionEvent] = []
        self._escalation_times: list[float] = []

    # -- policy ------------------------------------------------------------ #

    def _should_explain(self, result: InspectionResult) -> tuple[bool, str]:
        if self.explainer is None:
            return False, "no explainer configured"
        if not result.regions:
            return False, "no localised region to crop"
        if result.verdict == "PASS":
            return False, "verdict PASS"
        if result.verdict == "FAIL" and not self.policy.explain_on_fail:
            return False, "policy: explain_on_fail disabled"
        if result.verdict == "REVIEW":
            if not self.policy.explain_on_review:
                return False, "policy: explain_on_review disabled"
            if result.severity < self.policy.min_severity_for_explain:
                return False, f"severity {result.severity:.2f} below escalation floor"

        now = time.time()
        self._escalation_times = [t for t in self._escalation_times if now - t < 60.0]
        if len(self._escalation_times) >= self.policy.max_explanations_per_min:
            self.telemetry.throttled += 1
            return False, "throttled: escalation budget exhausted this minute"

        return True, f"verdict {result.verdict}, severity {result.severity:.2f}"

    # -- main loop --------------------------------------------------------- #

    def inspect(self, image: np.ndarray, *, part_id: str, compute_unit: str = "unknown") -> PartRecord:
        t_start = time.perf_counter()

        result = self.detector.inspect(image, compute_unit=compute_unit)
        self.telemetry.parts += 1
        self.telemetry.tier1_ms.append(result.latency_ms)

        explanation: Explanation | None = None
        do_explain, reason = self._should_explain(result)
        if do_explain:
            crop = result.regions[0].crop(image)
            explanation = self.explainer.explain(crop, part_name=self.part_name)
            self.telemetry.escalations += 1
            self.telemetry.tier2_ms.append(explanation.latency_ms)
            self._escalation_times.append(time.time())

        record = PartRecord(
            part_id=part_id, result=result, explanation=explanation,
            escalated=do_explain, escalation_reason=reason,
            total_ms=(time.perf_counter() - t_start) * 1000.0,
        )
        self.events.append(InspectionEvent(
            timestamp=time.time(),
            verdict=result.verdict,
            score=result.score,
            defect_type=explanation.defect_type if explanation else None,
            location=explanation.location if explanation else None,
            description=explanation.description if explanation else None,
        ))
        return record

    def close(self) -> None:
        if self.explainer is not None:
            self.explainer.close()
