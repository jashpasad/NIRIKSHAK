#!/usr/bin/env python3
"""
Generate a synthetic inspection dataset.

Why synthetic data ships in this repo: a reviewer must be able to clone, run one
command, and see the cascade work -- without a camera, a factory, or a licensed
dataset. Real validation is a different matter and is reported separately in
docs/BENCHMARKS.md against MVTec AD, which is the standard public benchmark for
this problem.

The generator models a machined/moulded rectangular component and injects
defects of the kinds that dominate real MSME reject bins:

    scratch          linear, high-frequency, low-contrast   -- the hard case
    dent             smooth low-frequency shading change
    contamination    blob of foreign material
    chip             missing material at an edge
    discolouration   broad low-contrast tint shift          -- the hardest case

Crucially, every good part also gets randomised lighting, position jitter, and
sensor noise. Without that the task is trivial and the numbers are a lie: any
detector separates a defect from a pixel-identical template. The nuisance
variation is what makes the benchmark mean something.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

SIZE = 384


def _base_part(rng: np.random.Generator) -> np.ndarray:
    """A brushed-metal-looking rectangular part with a bore and a stamped rib."""
    img = np.full((SIZE, SIZE, 3), 28, dtype=np.uint8)  # dark conveyor

    body_tone = rng.integers(138, 152)
    cv2.rectangle(img, (54, 74), (SIZE - 54, SIZE - 74),
                  (int(body_tone), int(body_tone) + 4, int(body_tone) + 9), -1)

    # Brushed finish: horizontal streaks, the texture most likely to be confused
    # with a scratch. Included deliberately.
    streaks = rng.normal(0, 7, (SIZE, 1)).astype(np.float32)
    streaks = cv2.GaussianBlur(streaks, (1, 9), 0)
    img = np.clip(img.astype(np.float32) + streaks[:, :, None], 0, 255).astype(np.uint8)

    cv2.circle(img, (SIZE // 2, SIZE // 2), 44, (96, 99, 104), -1)
    cv2.circle(img, (SIZE // 2, SIZE // 2), 44, (72, 75, 80), 3)
    cv2.rectangle(img, (86, 116), (SIZE - 86, 132), (124, 128, 134), -1)
    cv2.rectangle(img, (86, SIZE - 132), (SIZE - 86, SIZE - 116), (124, 128, 134), -1)
    for cx, cy in ((84, 104), (SIZE - 84, 104), (84, SIZE - 104), (SIZE - 84, SIZE - 104)):
        cv2.circle(img, (cx, cy), 9, (104, 107, 112), -1)
    return img


def _nuisance(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lighting gradient, sub-pixel jitter, slight rotation, sensor noise."""
    h, w = img.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx + np.sin(ang) * yy) / max(h, w)
    gain = 1.0 + rng.uniform(0.05, 0.16) * (grad - grad.mean())
    img = np.clip(img.astype(np.float32) * gain[:, :, None], 0, 255)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-1.4, 1.4), 1.0)
    M[0, 2] += rng.uniform(-4, 4)
    M[1, 2] += rng.uniform(-4, 4)
    img = cv2.warpAffine(img.astype(np.uint8), M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    img = np.clip(img.astype(np.float32) + rng.normal(0, 2.6, img.shape), 0, 255)
    return img.astype(np.uint8)


def good_part(rng: np.random.Generator) -> np.ndarray:
    return _nuisance(_base_part(rng), rng)


def defective_part(rng: np.random.Generator, kind: str) -> tuple[np.ndarray, str]:
    img = _base_part(rng)
    inner = (78, 98, SIZE - 78, SIZE - 98)

    if kind == "scratch":
        x1 = rng.integers(inner[0], inner[2] - 60)
        y1 = rng.integers(inner[1], inner[3])
        length = rng.integers(55, 130)
        angle = rng.uniform(-0.9, 0.9)
        x2 = int(np.clip(x1 + length * np.cos(angle), inner[0], inner[2]))
        y2 = int(np.clip(y1 + length * np.sin(angle), inner[1], inner[3]))
        overlay = img.copy()
        cv2.line(overlay, (int(x1), int(y1)), (x2, y2),
                 (int(rng.integers(178, 205)),) * 3, int(rng.integers(1, 3)), cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
        loc = (int(x1), int(y1))

    elif kind == "dent":
        cx = int(rng.integers(inner[0] + 30, inner[2] - 30))
        cy = int(rng.integers(inner[1] + 30, inner[3] - 30))
        r = int(rng.integers(20, 40))
        mask = np.zeros(img.shape[:2], np.float32)
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), r / 2.4)
        img = np.clip(img.astype(np.float32) * (1 - 0.30 * mask[:, :, None]), 0, 255).astype(np.uint8)
        loc = (cx, cy)

    elif kind == "contamination":
        cx = int(rng.integers(inner[0] + 24, inner[2] - 24))
        cy = int(rng.integers(inner[1] + 24, inner[3] - 24))
        pts = []
        for i in range(9):
            a = 2 * np.pi * i / 9
            rr = rng.integers(10, 24)
            pts.append([cx + rr * np.cos(a), cy + rr * np.sin(a)])
        cv2.fillPoly(img, [np.array(pts, np.int32)],
                     (int(rng.integers(38, 68)),) * 3)
        loc = (cx, cy)

    elif kind == "chip":
        edge = rng.integers(0, 4)
        if edge == 0:
            cx, cy = int(rng.integers(90, SIZE - 90)), 74
        elif edge == 1:
            cx, cy = int(rng.integers(90, SIZE - 90)), SIZE - 74
        elif edge == 2:
            cx, cy = 54, int(rng.integers(110, SIZE - 110))
        else:
            cx, cy = SIZE - 54, int(rng.integers(110, SIZE - 110))
        cv2.circle(img, (cx, cy), int(rng.integers(13, 24)), (28, 28, 28), -1)
        loc = (cx, cy)

    elif kind == "discolouration":
        cx = int(rng.integers(inner[0] + 40, inner[2] - 40))
        cy = int(rng.integers(inner[1] + 40, inner[3] - 40))
        r = int(rng.integers(38, 62))
        mask = np.zeros(img.shape[:2], np.float32)
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), r / 2.0)[:, :, None]
        tint = np.array([1.02, 0.92, 0.80], np.float32)
        img = np.clip(img.astype(np.float32) * (1 - mask) +
                      img.astype(np.float32) * tint * mask, 0, 255).astype(np.uint8)
        loc = (cx, cy)
    else:
        raise ValueError(f"unknown defect kind: {kind}")

    return _nuisance(img, rng), f"{loc[0]},{loc[1]}"


DEFECTS = ["scratch", "dent", "contamination", "chip", "discolouration"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/demo", type=Path)
    ap.add_argument("--good-train", type=int, default=24, help="enrolment set size")
    ap.add_argument("--good-test", type=int, default=40)
    ap.add_argument("--per-defect", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root: Path = args.out
    (root / "train" / "good").mkdir(parents=True, exist_ok=True)
    (root / "test" / "good").mkdir(parents=True, exist_ok=True)

    for i in range(args.good_train):
        cv2.imwrite(str(root / "train" / "good" / f"{i:03d}.png"),
                    cv2.cvtColor(good_part(rng), cv2.COLOR_RGB2BGR))
    for i in range(args.good_test):
        cv2.imwrite(str(root / "test" / "good" / f"{i:03d}.png"),
                    cv2.cvtColor(good_part(rng), cv2.COLOR_RGB2BGR))

    for kind in DEFECTS:
        d = root / "test" / kind
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_defect):
            img, loc = defective_part(rng, kind)
            cv2.imwrite(str(d / f"{i:03d}_at_{loc}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    total = args.good_train + args.good_test + len(DEFECTS) * args.per_defect
    print(f"Wrote {total} images to {root}")
    print(f"  enrolment (good): {args.good_train}")
    print(f"  test good:        {args.good_test}")
    print(f"  test defective:   {len(DEFECTS) * args.per_defect} "
          f"across {len(DEFECTS)} defect types")


if __name__ == "__main__":
    main()
