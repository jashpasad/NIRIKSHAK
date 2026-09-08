#!/usr/bin/env python3
"""
End-to-end demo: enrol from good parts, inspect a mixed batch, render results,
and produce a shift report.

Runs the full cascade. Tiers 2 and 3 degrade gracefully when GenieX is absent
(i.e. on any host that is not Windows-on-Snapdragon), so this script is
runnable everywhere and simply reports which tiers were live.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import cv2
import numpy as np

from nirikshak import (AnomalyDetector, ReferencePatchEmbedder, DefectExplainer,
                       InspectionCascade, CascadePolicy, ShiftReporter, compute_stats)
from nirikshak.tier3_reason.shift_report import export_events

VERDICT_COLOUR = {"PASS": (46, 160, 67), "REVIEW": (219, 154, 4), "FAIL": (218, 54, 51)}


def load_dir(d: Path) -> list[np.ndarray]:
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            for p in sorted(d.glob("*.png")) if cv2.imread(str(p)) is not None]


def render(image: np.ndarray, result, label: str) -> np.ndarray:
    """Original | heatmap overlay | verdict panel."""
    h, w = image.shape[:2]
    hm = (np.clip(result.heatmap, 0, 1) * 255).astype(np.uint8)
    hm_colour = cv2.applyColorMap(hm, cv2.COLORMAP_INFERNO)
    hm_colour = cv2.cvtColor(hm_colour, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.55, hm_colour, 0.45, 0)

    colour = VERDICT_COLOUR[result.verdict]
    for r in result.regions:
        cv2.rectangle(overlay, (r.x, r.y), (r.x + r.w, r.y + r.h), colour, 2)

    panel = np.full((h, 260, 3), 24, np.uint8)
    lines = [
        (label, 0.52, (235, 235, 235)),
        ("", 0, None),
        (result.verdict, 0.95, colour),
        ("", 0, None),
        (f"score    {result.score:.4f}", 0.44, (190, 190, 190)),
        (f"severity {result.severity:.2f}", 0.44, (190, 190, 190)),
        (f"regions  {len(result.regions)}", 0.44, (190, 190, 190)),
        (f"{result.latency_ms:.1f} ms", 0.44, (150, 190, 235)),
    ]
    y = 34
    for text, scale, col in lines:
        if col is not None and text:
            cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
        y += 34 if scale > 0.7 else 26

    bar = np.full((h, 4, 3), 24, np.uint8)
    return np.hstack([image, bar, overlay, bar, panel])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/demo"))
    ap.add_argument("--out", type=Path, default=Path("results/demo"))
    ap.add_argument("--n-good", type=int, default=20, help="good parts in the run")
    ap.add_argument("--enable-tier2", action="store_true",
                    help="load the VLM (requires GenieX on Windows ARM64)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # ---- enrol ---------------------------------------------------------- #
    train = load_dir(args.data / "train" / "good")
    embedder = ReferencePatchEmbedder()
    detector = AnomalyDetector(embedder)

    print("\n[1/4] Enrolling from good parts only -- no labels, no training")
    t0 = time.perf_counter()
    bank = detector.enrol(train)
    print(f"      {len(train)} samples -> {bank.size} bank entries "
          f"in {time.perf_counter()-t0:.1f} s")
    detector.save(args.out / "recipe.npz")

    # ---- build a realistic batch ---------------------------------------- #
    batch: list[tuple[np.ndarray, str]] = [
        (im, "good") for im in load_dir(args.data / "test" / "good")[:args.n_good]
    ]
    for d in sorted(p for p in (args.data / "test").iterdir()
                    if p.is_dir() and p.name != "good"):
        batch += [(im, d.name) for im in load_dir(d)[:3]]
    random.shuffle(batch)

    # ---- run the cascade ------------------------------------------------ #
    explainer = DefectExplainer() if args.enable_tier2 else None
    cascade = InspectionCascade(detector, explainer, CascadePolicy(), part_name="metal bracket")

    print(f"\n[2/4] Running cascade over {len(batch)} parts")
    print(f"      {'part':<22}{'truth':<16}{'verdict':<10}{'score':>8}{'ms':>8}  escalated")
    print(f"      {'-'*74}")

    saved = {}
    for i, (img, truth) in enumerate(batch):
        rec = cascade.inspect(img, part_id=f"P{i:04d}")
        r = rec.result
        mark = "yes" if rec.escalated else ""
        print(f"      P{i:04d}{'':<17}{truth:<16}{r.verdict:<10}"
              f"{r.score:>8.4f}{rec.total_ms:>8.1f}  {mark}")
        if rec.explanation:
            print(f"             -> {rec.explanation.defect_type}: "
                  f"{rec.explanation.description}")
        # Keep one rendered example per class for the figures.
        if truth not in saved:
            cv2.imwrite(str(args.out / f"inspect_{truth}.png"),
                        cv2.cvtColor(render(img, r, f"P{i:04d}  ({truth})"), cv2.COLOR_RGB2BGR))
            saved[truth] = True

    # ---- telemetry ------------------------------------------------------ #
    print("\n[3/4] Cascade telemetry")
    for k, v in cascade.telemetry.summary().items():
        print(f"      {k:<28}{v}")

    # ---- shift report --------------------------------------------------- #
    print("\n[4/4] Tier 3 shift report")
    stats = compute_stats(cascade.events)
    export_events(cascade.events, args.out / "events.jsonl")
    report = ShiftReporter().generate(
        stats, context="Fixture changed at 13:45. New operator on the second half of the shift.")
    (args.out / "shift_report.md").write_text(report, encoding="utf-8")
    print("\n" + report + "\n")
    print(f"Artifacts written to {args.out}/")


if __name__ == "__main__":
    main()
