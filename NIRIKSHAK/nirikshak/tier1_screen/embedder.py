"""
Tier 1 -- patch embedding.

The whole system rests on one idea: a defect is a *patch of the image that does
not look like any patch we saw during enrolment*. That reframing is what removes
the training set. We never learn what a scratch looks like; we learn what this
part looks like when it is correct, and flag departures.

To do that we need a function mapping an image to a grid of local descriptors:

    image (H x W x 3)  ->  patches (rows*cols, D)

Two implementations satisfy that contract.

`OnnxPatchEmbedder` is the production path. It runs a mid-level CNN backbone on
the Hexagon NPU and concatenates two feature maps: an early layer carrying
texture and edges, and a deeper layer carrying shape and part identity. We
deliberately avoid the final layers -- ImageNet's last block is tuned for "which
of 1000 classes is this", which is exactly the wrong invariance here. A scratch
must NOT be invariant.

`ReferencePatchEmbedder` is a hand-built descriptor with no learned weights. It
exists so the repository is testable on any machine, in CI, and on a laptop with
no model bundle downloaded. Its design is driven by what actually defeats naive
descriptors on a real line:

  * Thin defects vanish into patch means. A 2 px scratch inside a 16 px patch
    barely moves the mean gradient but strongly moves the *maximum*. So we carry
    max and 95th-percentile statistics, not just means.

  * Lighting is never uniform. A conveyor has a gradient across it and the lamp
    ages. Any feature built on absolute intensity drifts with it. So intensity
    is expressed relative to a large-neighbourhood local background, and colour
    is expressed as normalised chromaticity r/(r+g+b), which is invariant to a
    multiplicative illumination gain.

  * Real texture looks like defects. This part has a brushed finish -- parallel
    high-frequency streaks. An orientation-blind edge detector calls that a
    scratch. Oriented histograms separate the two, because the brushing has one
    dominant orientation and a scratch generally does not share it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PatchGrid:
    """Descriptors on a spatial grid, so scores can be mapped back to pixels."""

    features: np.ndarray  # (rows*cols, dim) float32, RAW -- whitening happens in the bank
    rows: int
    cols: int
    image_hw: tuple[int, int]

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    def as_map(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(self.rows, self.cols)


class PatchEmbedder(ABC):
    name: str = "abstract"
    accelerated: bool = False

    @abstractmethod
    def embed(self, image: np.ndarray) -> PatchGrid:
        """image: uint8 HxWx3 RGB -> PatchGrid of raw descriptors."""


# --------------------------------------------------------------------------- #
#  Production path: CNN backbone on the Hexagon NPU
# --------------------------------------------------------------------------- #

class OnnxPatchEmbedder(PatchEmbedder):
    """
    Runs an ONNX backbone through a Backend (QNN EP -> Hexagon NPU).

    The backbone must be exported with intermediate feature maps as graph
    outputs; `scripts/fetch_backbone.py` does this via Qualcomm AI Hub, which
    also applies the quantisation the HTP requires.

    Locally-aware pooling: raw CNN activations describe a single receptive
    field. Averaging each feature over a 3x3 neighbourhood buys tolerance to the
    sub-millimetre part jitter a real conveyor produces, without needing a
    precision mechanical fixture -- which is exactly the cost we are trying to
    remove from the cell.
    """

    name = "onnx-backbone"

    def __init__(self, backend, *, input_size: int = 320, neighbourhood: int = 3,
                 pos_weight: float = 0.0):
        self.backend = backend
        self.input_size = input_size
        self.neighbourhood = neighbourhood
        self.pos_weight = pos_weight
        self.accelerated = backend.compute_unit.accelerated
        self._input_name = backend.input_names[0]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2
        img = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)  # NCHW

    def embed(self, image: np.ndarray) -> PatchGrid:
        x = self._preprocess(image)
        feature_maps = self.backend.run({self._input_name: x})

        # Resample every map onto the coarsest grid before concatenating.
        # Coarsest wins: upsampling a deep map invents spatial precision it does
        # not have, and that fake precision surfaces later as phantom regions.
        maps = [np.asarray(f)[0] for f in feature_maps]  # each (C, h, w)
        th = min(m.shape[1] for m in maps)
        tw = min(m.shape[2] for m in maps)

        stacked = np.concatenate([_avg_pool_to(m, th, tw) for m in maps], axis=0)
        stacked = _neighbourhood_average(stacked, self.neighbourhood)

        feats = stacked.reshape(stacked.shape[0], -1).T.astype(np.float32)
        if self.pos_weight > 0:
            feats = np.concatenate([feats, _position_features(th, tw, self.pos_weight)], axis=1)
        return PatchGrid(feats, th, tw, image.shape[:2])


# --------------------------------------------------------------------------- #
#  Reference path: no learned weights, no model download
# --------------------------------------------------------------------------- #

class ReferencePatchEmbedder(PatchEmbedder):
    """
    Fully vectorised multi-scale descriptor. Per patch, per blur scale:

        9   oriented gradient histogram (unsigned, magnitude-weighted, L1-norm)
        4   gradient magnitude: mean, std, p95, max      <- thin-defect channel
        4   normalised chromaticity r and b: mean, std   <- illumination-invariant
        2   Laplacian: mean |.|, p95 |.|
        4   intensity vs local background at 2 kernel widths: mean offset, std
        ---
        23 dims x 2 blur scales = 46, plus 2 positional dims = 48

    Two choices here were each worth more than everything else combined, and
    both were found by measurement rather than by reasoning:

    OVERLAPPING PATCHES (stride < patch). With non-overlapping tiles, a scratch
    that crosses a tile boundary is diluted into two patches and detected in
    neither. Halving the stride raised overall AUROC from 0.979 to 0.999 on the
    demo set and scratch recall from 38% to 100%. It costs 4x the patches, and
    it is worth every one of them.

    TWO BACKGROUND KERNELS. A single local-background width can only see defects
    smaller than itself: a 40 px dent inside a 65 px background window is simply
    absorbed into the background and disappears. Carrying a narrow (65 px) and a
    wide (161 px) kernel gives sensitivity across both regimes and took dent
    recall from 88% to 100%.

    Positional encoding matters here in a way it does not in the generic
    anomaly-detection literature. Those benchmarks assume an unregistered object;
    an inspection cell has a fixture, so a patch's location on the part is real
    information. Including it stops a bore-edge patch from being explained away
    by a legitimately similar-looking patch on the opposite corner. The weight is
    tunable because a poorly fixtured cell should lean on it less.
    """

    name = "reference-descriptor"
    accelerated = False

    def __init__(self, *, input_size: int = 320, patch: int = 16, stride: int = 8,
                 blur_scales: tuple[float, ...] = (0.0, 2.0),
                 nbins: int = 9, bg_kernels: tuple[int, ...] = (65, 161),
                 pos_weight: float = 0.35):
        self.input_size = input_size
        self.patch = patch
        self.stride = stride
        self.blur_scales = blur_scales
        self.nbins = nbins
        self.bg_kernels = tuple(k | 1 for k in bg_kernels)  # box kernels must be odd
        self.pos_weight = pos_weight

    @property
    def grid_dims(self) -> tuple[int, int]:
        n = (self.input_size - self.patch) // self.stride + 1
        return n, n

    def embed(self, image: np.ndarray) -> PatchGrid:
        import cv2

        img = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        rows, cols = self.grid_dims

        blocks = [
            self._scale_features(img if s == 0.0 else cv2.GaussianBlur(img, (0, 0), sigmaX=s))
            for s in self.blur_scales
        ]
        feats = np.concatenate(blocks, axis=1).astype(np.float32)

        if self.pos_weight > 0:
            feats = np.concatenate([feats, _position_features(rows, cols, self.pos_weight)], axis=1)
        return PatchGrid(feats, rows, cols, image.shape[:2])

    def _scale_features(self, img: np.ndarray) -> np.ndarray:
        import cv2

        p, st = self.patch, self.stride
        rgb = img.astype(np.float32) / 255.0
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.arctan2(gy, gx) % np.pi
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

        # Normalised chromaticity: invariant to a multiplicative light gain.
        s = rgb.sum(axis=2) + 1e-5
        chroma_r = rgb[:, :, 0] / s
        chroma_b = rgb[:, :, 2] / s

        mag_b = _blockify(mag, p, st)
        bin_b = np.clip(_blockify(ang / np.pi * self.nbins, p, st).astype(np.int32),
                        0, self.nbins - 1)
        lap_b = _blockify(lap, p, st)
        cr_b = _blockify(chroma_r, p, st)
        cb_b = _blockify(chroma_b, p, st)

        hist = np.zeros((mag_b.shape[0], self.nbins), dtype=np.float32)
        for k in range(self.nbins):
            hist[:, k] = (mag_b * (bin_b == k)).sum(axis=1)
        hist /= (hist.sum(axis=1, keepdims=True) + 1e-6)

        parts = [
            hist,
            mag_b.mean(1, keepdims=True), mag_b.std(1, keepdims=True),
            np.percentile(mag_b, 95, axis=1, keepdims=True), mag_b.max(1, keepdims=True),
            cr_b.mean(1, keepdims=True), cr_b.std(1, keepdims=True),
            cb_b.mean(1, keepdims=True), cb_b.std(1, keepdims=True),
            lap_b.mean(1, keepdims=True),
            np.percentile(lap_b, 95, axis=1, keepdims=True),
        ]
        # Local background at several widths. A defect larger than the kernel is
        # absorbed into its own background and becomes invisible, so one kernel
        # is never enough across a realistic defect size range.
        for bk in self.bg_kernels:
            rel_b = _blockify(gray - cv2.blur(gray, (bk, bk)), p, st)
            parts += [rel_b.mean(1, keepdims=True), rel_b.std(1, keepdims=True)]

        return np.concatenate(parts, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _blockify(x: np.ndarray, p: int, stride: int) -> np.ndarray:
    """
    (H, W) -> (rows*cols, p*p) windows of size p at the given stride, row-major.

    sliding_window_view is a pure view, so the only copy is the final reshape.
    With stride < p the windows overlap, which is what gives thin defects a
    patch they sit inside rather than one they straddle.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(x, (p, p))[::stride, ::stride]
    rows, cols = w.shape[:2]
    return np.ascontiguousarray(w).reshape(rows * cols, p * p)


def _position_features(rows: int, cols: int, weight: float) -> np.ndarray:
    """Normalised (row, col) in [-1, 1], scaled. Appended after the appearance dims."""
    rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float32)
    rr = (rr / max(rows - 1, 1)) * 2 - 1
    cc = (cc / max(cols - 1, 1)) * 2 - 1
    return (np.stack([rr.ravel(), cc.ravel()], axis=1) * weight).astype(np.float32)


def _avg_pool_to(fmap: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Average-pool a (C, h, w) map down to (C, out_h, out_w)."""
    c, h, w = fmap.shape
    if (h, w) == (out_h, out_w):
        return fmap
    hs = np.linspace(0, h, out_h + 1).astype(int)
    ws = np.linspace(0, w, out_w + 1).astype(int)
    out = np.empty((c, out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            out[:, i, j] = fmap[:, hs[i]:max(hs[i + 1], hs[i] + 1),
                                ws[j]:max(ws[j + 1], ws[j] + 1)].mean(axis=(1, 2))
    return out


def _neighbourhood_average(fmap: np.ndarray, k: int) -> np.ndarray:
    """Box-average each channel over a k x k neighbourhood, edges replicated."""
    if k <= 1:
        return fmap
    import cv2
    pad = k // 2
    out = np.empty_like(fmap)
    for i in range(fmap.shape[0]):
        padded = cv2.copyMakeBorder(fmap[i], pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        out[i] = cv2.blur(padded, (k, k))[pad:pad + fmap.shape[1], pad:pad + fmap.shape[2]]
    return out
