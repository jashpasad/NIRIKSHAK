#!/usr/bin/env python3
"""
Evaluate Tier 1 on a dataset laid out as:

    root/train/good/*.png          enrolment set
    root/test/good/*.png           known-good
    root/test/<defect_type>/*.png  known-defective

Reports image-level AUROC (threshold-free separability), and then the two
numbers a factory actually cares about, which are NOT AUROC:

    escape rate   -- defective parts marked PASS. Every one reaches a customer.
    false reject  -- good parts marked FAIL. Every one is scrapped revenue.

REVIEW is counted as neither. That is the point of the middle band: a part in
review has not been decided yet, it has been escalated. Papers that collapse
this to a binary make the escape rate look better than it is.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from nirikshak import AnomalyDetector, ReferencePatchEmbedder


def load_dir(d: Path) -> list[np.ndarray]:
    imgs = []
    for p in sorted(d.glob("*.png")):
        im = cv2.imread(str(p))
        if im is not None:
            imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return imgs


def auroc(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUROC (Mann-Whitney U), tie-corrected."""
    if not pos or not neg:
        return float("nan")
    scores = np.array(neg + pos, dtype=np.float64)
    labels = np.array([0] * len(neg) + [1] * len(pos))
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.sort(scores)
    i = 0
    while i < len(s):  # average ranks within tie groups
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/demo"))
    ap.add_argument("--coreset", type=float, default=0.05)
    ap.add_argument("--knn", type=int, default=1)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--line-yield", type=float, default=97.0,
                    help="Assumed share of good parts on the real line. The test set is "
                         "deliberately balanced for statistics, but escalation economics "
                         "must be computed at the mix the cell will actually see.")
    args = ap.parse_args()

    train = load_dir(args.data / "train" / "good")
    if not train:
        raise SystemExit(f"No enrolment images in {args.data/'train'/'good'}. "
                         "Run scripts/make_demo_data.py first.")

    embedder = ReferencePatchEmbedder()
    det = AnomalyDetector(embedder, knn=args.knn)

    t0 = time.perf_counter()
    bank = det.enrol(train, coreset_ratio=args.coreset)
    enrol_s = time.perf_counter() - t0

    print(f"\n{'='*66}")
    print("ENROLMENT")
    print(f"{'='*66}")
    print(f"  embedder            {embedder.name}  (dim={bank.vectors.shape[1]})")
    print(f"  good samples        {len(train)}")
    print(f"  wall time           {enrol_s:.2f} s")
    print(f"  memory bank         {bank.size} descriptors "
          f"({bank.vectors.nbytes/1024:.0f} KB)")
    c = bank.calibration
    print(f"  normal score        mean {c.normal_mean:.4f}  sd {c.normal_std:.4f}")
    print(f"  PASS below          {c.pass_below:.4f}")
    print(f"  FAIL above          {c.fail_above:.4f}")

    good_scores, good_verdicts, lat = [], [], []
    for im in load_dir(args.data / "test" / "good"):
        r = det.inspect(im)
        good_scores.append(r.score)
        good_verdicts.append(r.verdict)
        lat.append(r.latency_ms)

    defect_dirs = sorted(p for p in (args.data / "test").iterdir()
                         if p.is_dir() and p.name != "good")

    print(f"\n{'='*66}")
    print("DETECTION BY DEFECT TYPE")
    print(f"{'='*66}")
    print(f"  {'defect type':<18}{'n':>4}{'AUROC':>9}{'caught':>9}{'escaped':>9}{'localised':>11}")
    print(f"  {'-'*60}")

    all_def_scores: list[float] = []
    per_type = {}
    for d in defect_dirs:
        s, v, loc_ok = [], [], 0
        for im in load_dir(d):
            r = det.inspect(im)
            s.append(r.score)
            v.append(r.verdict)
            lat.append(r.latency_ms)
            if r.regions:
                loc_ok += 1
        if not s:
            continue
        all_def_scores.extend(s)
        caught = sum(1 for x in v if x in ("FAIL", "REVIEW"))
        escaped = sum(1 for x in v if x == "PASS")
        a = auroc(s, good_scores)
        per_type[d.name] = {"n": len(s), "auroc": round(a, 4),
                            "caught": caught, "escaped": escaped, "localised": loc_ok}
        print(f"  {d.name:<18}{len(s):>4}{a:>9.3f}{caught:>9}{escaped:>9}{loc_ok:>11}")

    overall = auroc(all_def_scores, good_scores)
    n_def = len(all_def_scores)
    escapes = sum(1 for d in per_type.values() for _ in range(d["escaped"]))
    false_rejects = sum(1 for v in good_verdicts if v == "FAIL")
    reviews_good = sum(1 for v in good_verdicts if v == "REVIEW")

    print(f"  {'-'*60}")
    print(f"  {'OVERALL':<18}{n_def:>4}{overall:>9.3f}")

    print(f"\n{'='*66}")
    print("LINE ECONOMICS")
    print(f"{'='*66}")
    print(f"  escape rate         {100*escapes/max(n_def,1):.1f}%  "
          f"({escapes}/{n_def} defects marked PASS)")
    print(f"  false reject rate   {100*false_rejects/max(len(good_verdicts),1):.1f}%  "
          f"({false_rejects}/{len(good_verdicts)} good parts marked FAIL)")
    print(f"  good -> review      {100*reviews_good/max(len(good_verdicts),1):.1f}%  "
          f"(operator sees these; cost is attention, not scrap)")

    # Escalation economics must be computed at the mix a real line sees, not at
    # the balanced mix used for AUROC. A 50/50 test set makes the cascade look
    # far worse than it is, because half its parts are defective.
    y = args.line_yield / 100.0
    p_esc_good = (reviews_good + false_rejects) / max(len(good_verdicts), 1)
    p_esc_def = (n_def - escapes) / max(n_def, 1)
    esc_rate = y * p_esc_good + (1 - y) * p_esc_def

    t1 = float(np.mean(lat))
    t2 = 400.0  # measured Tier-2 VLM latency; see docs/BENCHMARKS.md
    eff = t1 + esc_rate * t2

    print(f"\n  Tier 1 latency      {t1:.1f} ms mean, {np.percentile(lat,95):.1f} ms p95"
          f"  ({embedder.name}, CPU reference)")
    print(f"  assumed line yield  {args.line_yield:.0f}% good")
    print(f"    P(escalate|good)  {100*p_esc_good:.1f}%")
    print(f"    P(escalate|bad)   {100*p_esc_def:.1f}%")
    print(f"  Tier 2 escalation   {100*esc_rate:.1f}% of parts at that mix")
    print(f"  effective cost      {eff:.0f} ms/part vs {t2:.0f} ms if a VLM ran on every part")
    print(f"  speed-up from gating  {t2/eff:.1f}x")

    # Report the window of thresholds that are simultaneously safe, so a
    # deployer can see how much margin the separation actually has.
    gmax, dmin = max(good_scores), min(all_def_scores)
    if gmax < dmin:
        lo = (gmax - c.normal_mean) / c.normal_std
        hi = (dmin - c.normal_mean) / c.normal_std
        print(f"\n  separation gap      good max {gmax:.4f} < defect min {dmin:.4f}")
        print(f"  safe pass_sigma     [{lo:.2f}, {hi:.2f}]  (configured: "
              f"{(c.pass_below-c.normal_mean)/c.normal_std:.2f})")
    else:
        print(f"\n  separation gap      NONE -- good max {gmax:.4f} >= defect min {dmin:.4f}; "
              "no threshold separates these classes")
    print()

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "embedder": embedder.name,
            "enrolment": {"images": len(train), "seconds": round(enrol_s, 3),
                          "bank_entries": bank.size},
            "overall_auroc": round(overall, 4),
            "per_defect": per_type,
            "escape_rate_pct": round(100*escapes/max(n_def,1), 2),
            "false_reject_pct": round(100*false_rejects/max(len(good_verdicts),1), 2),
            "tier1_mean_ms": round(t1, 2),
            "escalation_rate_pct": round(100*esc_rate, 2),
        }, indent=2))
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
