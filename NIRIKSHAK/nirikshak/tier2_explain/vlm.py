"""
Tier 2 -- the explanation.

Tier 1 says "something at (412, 260) is 3.1 sigma from normal." That is a number
an engineer can act on and an operator cannot. Tier 2 converts it into a
sentence, and only for the parts that need one.

This tier runs a vision-language model on the Hexagon NPU through GenieX. It is
roughly 40x the cost of Tier 1, which is precisely why it is gated: on a line at
95% yield with a 3% review band, fewer than one part in twelve ever reaches it.
That gate is the difference between a laptop keeping up with a conveyor and not.

Design decisions worth defending:

* We send a *crop*, not the full frame. The crop is chosen by Tier 1's heatmap.
  A VLM asked to find a 4 mm scratch in a 12 MP frame will confabulate; a VLM
  shown the scratch will describe it. We use the cheap model to aim the
  expensive one.

* We ask for constrained JSON, not prose. Free-form output cannot be logged,
  counted, or trended, and an inspection record that cannot be trended is
  decoration.

* The model is never the decision-maker. Tier 1's calibrated distance decides
  pass/fail. Tier 2 supplies vocabulary. Letting a generative model adjudicate
  a physical accept/reject would be indefensible on an audited line, and we do
  not do it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_VLM = "ai-hub-models/Qwen3-VL-4B-Instruct"

# Kept deliberately short. Every token here is prefill the NPU pays for on every
# escalation, and a long taxonomy makes the model hedge across categories.
SYSTEM_PROMPT = """You are a visual quality inspector on a manufacturing line.
You are shown a close-up crop of a region that an anomaly detector has already \
flagged as differing from known-good samples of this part. The anomaly is real; \
your job is to name and describe it, not to re-decide whether it exists.

Reply with ONLY a JSON object, no other text:
{"defect_type": "<one of: scratch, dent, crack, chip, contamination, discolouration, \
missing_feature, deformation, print_defect, assembly_error, other>",
 "description": "<one factual sentence, max 25 words>",
 "location": "<where on the part, e.g. 'lower-left edge'>",
 "confidence": <0.0-1.0>}"""


@dataclass
class Explanation:
    defect_type: str
    description: str
    location: str
    confidence: float
    latency_ms: float
    raw: str = ""

    @classmethod
    def unavailable(cls, reason: str) -> "Explanation":
        return cls("unclassified", f"Tier 2 unavailable: {reason}", "unknown", 0.0, 0.0)


class DefectExplainer:
    """Wraps a GenieX VLM session. Loads lazily -- model load is ~5-15 s."""

    def __init__(self, model_id: str = DEFAULT_VLM, *, device_map: str = "qairt",
                 max_new_tokens: int = 120):
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
            self._unavailable_reason = (
                "geniex not installed (expected on non-Snapdragon hosts; "
                "install with `pip install geniex` on Windows ARM64)"
            )
            log.warning("Tier 2 disabled: %s", self._unavailable_reason)
            return
        try:
            t0 = time.perf_counter()
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=self.device_map,
            )
            log.info("Loaded %s on '%s' in %.1f s",
                     self.model_id, self.device_map, time.perf_counter() - t0)
        except Exception as exc:  # pragma: no cover - hardware dependent
            self._unavailable_reason = str(exc)
            log.error("Tier 2 model load failed: %s", exc)

    def explain(self, crop: np.ndarray, *, part_name: str = "part",
                tmp_dir: str | Path = ".nirikshak_tmp") -> Explanation:
        self._ensure()
        if self._model is None:
            return Explanation.unavailable(self._unavailable_reason or "unknown")

        import cv2
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        crop_path = (tmp_dir / "tier2_crop.png").resolve()
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": str(crop_path)},
                {"type": "text", "text": f"Part under inspection: {part_name}."},
            ]},
        ]
        prompt = self._model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        t0 = time.perf_counter()
        out = self._model.generate(
            prompt, images=[str(crop_path)], max_new_tokens=self.max_new_tokens,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        return _parse(out.text, latency)

    def close(self) -> None:
        if self._model is not None:
            self._model.close()
            self._model = None


def _parse(text: str, latency_ms: float) -> Explanation:
    """
    Small models wrap JSON in fences or prose often enough that a bare
    json.loads is not good enough for a production loop. We extract the first
    balanced object and degrade gracefully rather than crashing the line.
    """
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group(0))
            return Explanation(
                defect_type=str(d.get("defect_type", "other")),
                description=str(d.get("description", "")).strip(),
                location=str(d.get("location", "unknown")),
                confidence=float(d.get("confidence", 0.5)),
                latency_ms=latency_ms,
                raw=text,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    log.debug("Tier 2 returned unparseable output: %r", text[:200])
    return Explanation("other", text.strip()[:160] or "no description returned",
                       "unknown", 0.3, latency_ms, raw=text)
