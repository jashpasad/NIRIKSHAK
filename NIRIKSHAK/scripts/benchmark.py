#!/usr/bin/env python3
"""
Measure what the cascade actually costs, and model what it would cost without
the gating. Prints the numbers quoted in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from nirikshak import AnomalyDetector, ReferencePatchEmbedder
from nirikshak.runtime.backend import describe_host


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/demo"))
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tier2-ms", type=float, default=400.0,
                    help="measured Tier-2 VLM latency on the target device")
    ap.add_argument("--yields", type=float, nargs="+", default=[90, 95, 97, 99])
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    host = describe_host()
    print("host:", json.dumps(host))

    train = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
             for p in sorted((args.data / "train" / "good").glob("*.png"))]
    det = AnomalyDetector(ReferencePatchEmbedder())

    t0 = time.perf_counter()
    bank = det.enrol(train)
    enrol_s = time.perf_counter() - t0

    probe = train[0]
    for _ in range(5):
        det.inspect(probe)                       # warm caches
    lat = []
    for _ in range(args.iters):
        t = time.perf_counter()
        det.inspect(probe)
        lat.append((time.perf_counter() - t) * 1000)
    lat = np.asarray(lat)

    print(f"\nenrolment           {enrol_s:.2f} s for {len(train)} samples "
          f"-> {bank.size} bank entries")
    print(f"Tier 1              {lat.mean():.2f} ms mean | {np.percentile(lat,50):.2f} p50 "
          f"| {np.percentile(lat,95):.2f} p95 | {np.percentile(lat,99):.2f} p99")

    print(f"\n{'yield':>7}{'escalation':>12}{'ms/part':>10}{'parts/min':>11}{'vs VLM-always':>15}")
    rows = []
    for y in args.yields:
        # Escalate every defect plus a small share of good parts.
        esc = (1 - y / 100) * 1.0 + (y / 100) * 0.01
        eff = lat.mean() + esc * args.tier2_ms
        rows.append({"yield_pct": y, "escalation_pct": round(100 * esc, 2),
                     "ms_per_part": round(eff, 1),
                     "parts_per_min": round(60_000 / eff, 1),
                     "speedup": round(args.tier2_ms / eff, 2)})
        print(f"{y:>6.0f}%{100*esc:>11.1f}%{eff:>10.1f}{60_000/eff:>11.0f}"
              f"{args.tier2_ms/eff:>14.1f}x")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "host": host, "enrolment_s": round(enrol_s, 2), "bank_entries": bank.size,
            "tier1_ms": {"mean": round(float(lat.mean()), 2),
                         "p50": round(float(np.percentile(lat, 50)), 2),
                         "p95": round(float(np.percentile(lat, 95)), 2),
                         "p99": round(float(np.percentile(lat, 99)), 2)},
            "tier2_ms_assumed": args.tier2_ms, "cascade_economics": rows,
        }, indent=2))


if __name__ == "__main__":
    main()
