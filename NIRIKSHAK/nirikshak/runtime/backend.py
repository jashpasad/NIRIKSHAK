"""
Execution backend abstraction for NIRIKSHAK.

One rule governs this module: the pipeline never knows which silicon it is on.
It asks for a Backend, gets one, and calls .run(). Everything Snapdragon-specific
is confined here so that the same source tree develops on an x86 workstation and
deploys unchanged to a Snapdragon-powered HP PC.

Backend selection order on Windows-on-Snapdragon:
    1. QNNExecutionProvider   -> Hexagon NPU (HTP). The target path.
    2. DmlExecutionProvider   -> Adreno GPU. Fallback if a graph has an op the
                                 HTP backend cannot partition.
    3. CPUExecutionProvider   -> Oryon CPU. Development / correctness reference.

We deliberately do NOT hide which one was chosen. A quality-inspection cell that
silently drops from 10 ms/part to 400 ms/part is a broken cell, so the resolved
provider is surfaced in telemetry and printed at startup.
"""

from __future__ import annotations

import logging
import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Compute-unit description
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ComputeUnit:
    """What we actually ended up running on."""

    name: str                 # "Hexagon NPU" | "Adreno GPU" | "Oryon CPU" | "x86 CPU"
    provider: str             # the ONNX Runtime execution provider string
    accelerated: bool         # True only when the NPU took the graph
    notes: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        tag = "accelerated" if self.accelerated else "reference"
        return f"{self.name} [{self.provider}] ({tag})"


@dataclass
class InferenceStats:
    """Rolling latency accounting. Cheap enough to leave on in production."""

    count: int = 0
    total_ms: float = 0.0
    samples: list[float] = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        # Bounded reservoir: we only ever need percentiles, not the full history.
        if len(self.samples) < 4096:
            self.samples.append(ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        return float(np.percentile(np.asarray(self.samples), p))

    def summary(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.percentile(50), 2),
            "p95_ms": round(self.percentile(95), 2),
            "p99_ms": round(self.percentile(99), 2),
        }


# --------------------------------------------------------------------------- #
#  Backend interface
# --------------------------------------------------------------------------- #

class Backend(ABC):
    """A loaded model that can be run on some compute unit."""

    def __init__(self, model_path: str | Path, compute_unit: ComputeUnit):
        self.model_path = Path(model_path)
        self.compute_unit = compute_unit
        self.stats = InferenceStats()

    @abstractmethod
    def _forward(self, inputs: dict[str, np.ndarray]) -> Sequence[np.ndarray]:
        ...

    def run(self, inputs: dict[str, np.ndarray]) -> Sequence[np.ndarray]:
        t0 = time.perf_counter()
        out = self._forward(inputs)
        self.stats.record((time.perf_counter() - t0) * 1000.0)
        return out

    def close(self) -> None:  # pragma: no cover - most backends need no teardown
        pass


class OnnxBackend(Backend):
    """
    ONNX Runtime backend.

    On Snapdragon this is the QNN Execution Provider, which lowers the graph to
    the Hexagon Tensor Processor. The QNN EP requires a statically-shaped,
    quantized graph -- see docs/DEPLOYMENT.md for how the enrolment backbone is
    prepared with Qualcomm AI Hub before it lands here.
    """

    # Preference order. First provider that is both requested and available wins.
    _PREFERENCE: tuple[tuple[str, str, bool], ...] = (
        ("QNNExecutionProvider", "Hexagon NPU", True),
        ("DmlExecutionProvider", "Adreno GPU", False),
        ("CPUExecutionProvider", "CPU", False),
    )

    def __init__(
        self,
        model_path: str | Path,
        *,
        prefer: str | None = None,
        htp_performance_mode: str = "burst",
        context_cache: str | Path | None = None,
    ):
        import onnxruntime as ort  # imported lazily so docs build without ORT

        self._ort = ort
        available = set(ort.get_available_providers())
        log.debug("ONNX Runtime providers available: %s", sorted(available))

        chosen: tuple[str, str, bool] | None = None
        for provider, label, accel in self._PREFERENCE:
            if prefer and provider != prefer:
                continue
            if provider in available:
                chosen = (provider, label, accel)
                break
        if chosen is None:
            chosen = ("CPUExecutionProvider", "CPU", False)

        provider, label, accel = chosen
        if label == "CPU":
            label = "Oryon CPU" if platform.machine().lower() in {"arm64", "aarch64"} else "x86 CPU"

        provider_options: dict[str, Any] = {}
        if provider == "QNNExecutionProvider":
            # backend_path selects the HTP (NPU) backend rather than the CPU
            # reference backend inside QNN. htp_performance_mode=burst is what
            # you want for a line-rate inspection loop; 'sustained_high_performance'
            # is the right choice for the Tier-3 batch report at end of shift.
            provider_options = {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": htp_performance_mode,
                "htp_graph_finalization_optimization_mode": "3",
                "enable_htp_fp16_precision": "1",
            }
            if context_cache is not None:
                # Caching the compiled QNN context binary turns a ~20 s cold
                # graph finalization into a <1 s warm start. Matters a lot for
                # an operator who power-cycles the cell every shift.
                provider_options["qnn_context_cache_enable"] = "1"
                provider_options["qnn_context_cache_path"] = str(context_cache)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=[provider],
            provider_options=[provider_options] if provider_options else None,
        )

        # Trust but verify: ORT silently falls back if the EP rejects the graph.
        actually_used = self.session.get_providers()[0]
        if actually_used != provider:
            log.warning(
                "Requested %s but ONNX Runtime resolved to %s -- the graph was "
                "not accepted by the requested execution provider.",
                provider, actually_used,
            )
            accel = actually_used == "QNNExecutionProvider"
            label = {
                "QNNExecutionProvider": "Hexagon NPU",
                "DmlExecutionProvider": "Adreno GPU",
            }.get(actually_used, label)
            provider = actually_used

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        super().__init__(
            model_path,
            ComputeUnit(
                name=label,
                provider=provider,
                accelerated=accel,
                notes=f"inputs={self.input_names} outputs={self.output_names}",
            ),
        )

    def _forward(self, inputs: dict[str, np.ndarray]) -> Sequence[np.ndarray]:
        return self.session.run(self.output_names, inputs)


def describe_host() -> dict[str, str]:
    """Best-effort description of the machine we are running on, for telemetry."""
    info = {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "system": platform.system(),
        "python": platform.python_version(),
    }
    try:  # pragma: no cover - Windows-only path
        import subprocess
        if platform.system() == "Windows":
            out = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if len(lines) > 1:
                info["processor"] = lines[1]
    except Exception:
        pass
    return info
