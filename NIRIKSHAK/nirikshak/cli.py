"""Command-line interface: enrol, inspect, serve, bench."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def _load_images(path: Path) -> list:
    import cv2
    paths = sorted(p for p in path.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None:
            out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return out


def _make_embedder(args):
    """Pick the production NPU path when a backbone is supplied, else the reference."""
    from .tier1_screen.embedder import ReferencePatchEmbedder, OnnxPatchEmbedder
    if args.backbone:
        from .runtime.backend import OnnxBackend
        backend = OnnxBackend(args.backbone, context_cache=args.context_cache)
        print(f"compute: {backend.compute_unit}", file=sys.stderr)
        return OnnxPatchEmbedder(backend), backend.compute_unit.name
    return ReferencePatchEmbedder(), "CPU reference"


def cmd_enrol(args) -> int:
    from .tier1_screen.detector import AnomalyDetector
    images = _load_images(args.good)
    if not images:
        print(f"No images found in {args.good}", file=sys.stderr)
        return 1
    embedder, unit = _make_embedder(args)
    det = AnomalyDetector(embedder)
    bank = det.enrol(images, coreset_ratio=args.coreset)
    det.save(args.out)
    c = bank.calibration
    print(json.dumps({
        "recipe": str(args.out), "compute": unit, "samples": len(images),
        "bank_entries": bank.size,
        "pass_below": round(c.pass_below, 5), "fail_above": round(c.fail_above, 5),
    }, indent=2))
    return 0


def cmd_inspect(args) -> int:
    import cv2
    from .tier1_screen.detector import AnomalyDetector
    embedder, unit = _make_embedder(args)
    det = AnomalyDetector.load(args.recipe, embedder)
    im = cv2.imread(str(args.image))
    if im is None:
        print(f"Cannot read {args.image}", file=sys.stderr)
        return 1
    r = det.inspect(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), compute_unit=unit)
    print(json.dumps(r.to_dict(), indent=2))
    if args.heatmap:
        hm = cv2.applyColorMap((np.clip(r.heatmap, 0, 1) * 255).astype(np.uint8),
                               cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(args.heatmap), cv2.addWeighted(im, 0.55, hm, 0.45, 0))
    return 0 if r.verdict == "PASS" else 2   # non-zero exit lets shell scripts gate on it


def cmd_serve(args) -> int:
    from app.server import serve
    serve(recipe=args.recipe, backbone=args.backbone, host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="nirikshak", description=__doc__)
    ap.add_argument("--backbone", type=Path, default=None,
                    help="ONNX backbone for the NPU path; omit to use the reference embedder")
    ap.add_argument("--context-cache", type=Path, default=None,
                    help="QNN context binary cache (turns ~20 s cold start into <1 s)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enrol", help="build a recipe from a folder of good parts")
    e.add_argument("good", type=Path)
    e.add_argument("-o", "--out", type=Path, default=Path("recipe.npz"))
    e.add_argument("--coreset", type=float, default=0.05)
    e.set_defaults(func=cmd_enrol)

    i = sub.add_parser("inspect", help="inspect one image against a recipe")
    i.add_argument("recipe", type=Path)
    i.add_argument("image", type=Path)
    i.add_argument("--heatmap", type=Path, default=None)
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("serve", help="run the local inspection server + UI")
    s.add_argument("recipe", type=Path)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
