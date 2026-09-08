"""
Tests that assert behaviour, not implementation.

Each one encodes a property that, if it broke silently, would produce a system
that still runs and still prints verdicts while being wrong -- which is the only
failure mode that actually matters on a production line.
"""

import numpy as np
import pytest

from nirikshak import AnomalyDetector, ReferencePatchEmbedder, InspectionCascade, CascadePolicy
from nirikshak.tier1_screen.memory_bank import greedy_coreset, Whitener
from nirikshak.tier3_reason.shift_report import InspectionEvent, compute_stats


def _part(rng, defect=False):
    img = np.full((256, 256, 3), 140, np.uint8)
    img[40:216, 40:216] = 165
    img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    if defect:
        img[120:132, 90:170] = 235          # a bright bar: unambiguous anomaly
    return img


@pytest.fixture(scope="module")
def enrolled():
    rng = np.random.default_rng(0)
    det = AnomalyDetector(ReferencePatchEmbedder(input_size=192, patch=16, stride=8))
    det.enrol([_part(rng) for _ in range(8)])
    return det


def test_good_parts_pass(enrolled):
    rng = np.random.default_rng(99)
    verdicts = [enrolled.inspect(_part(rng)).verdict for _ in range(10)]
    assert verdicts.count("FAIL") == 0, f"false rejects on good parts: {verdicts}"


def test_defects_are_caught_and_localised(enrolled):
    rng = np.random.default_rng(123)
    r = enrolled.inspect(_part(rng, defect=True))
    assert r.verdict in ("FAIL", "REVIEW")
    assert r.regions, "a caught defect must be localised, or the operator cannot act on it"
    # The synthetic defect sits at rows 120-132; the top region must overlap it.
    top = r.regions[0]
    assert top.y <= 132 and top.y + top.h >= 120


def test_enrolment_refuses_too_few_samples():
    det = AnomalyDetector(ReferencePatchEmbedder(input_size=192))
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="at least 3"):
        det.enrol([_part(rng), _part(rng)])


def test_recipe_roundtrip(enrolled, tmp_path):
    rng = np.random.default_rng(5)
    img = _part(rng, defect=True)
    before = enrolled.inspect(img)
    path = tmp_path / "r.npz"
    enrolled.save(path)
    reloaded = AnomalyDetector.load(path, enrolled.embedder)
    after = reloaded.inspect(img)
    assert after.verdict == before.verdict
    assert abs(after.score - before.score) < 1e-5


def test_recipe_rejects_mismatched_embedder(enrolled, tmp_path):
    """Descriptors from different embedders are not comparable. Loading one into
    the other must fail loudly rather than silently produce nonsense verdicts."""
    from nirikshak.tier1_screen.embedder import PatchEmbedder

    class Other(PatchEmbedder):
        name = "something-else"
        def embed(self, image):  # pragma: no cover
            raise NotImplementedError

    path = tmp_path / "r.npz"
    enrolled.save(path)
    with pytest.raises(ValueError, match="enrolled with embedder"):
        AnomalyDetector.load(path, Other())


def test_coreset_keeps_extremes():
    """The point of k-center over random sampling: outliers must survive."""
    rng = np.random.default_rng(0)
    blob = rng.normal(0, 0.01, (400, 8))
    outliers = np.array([[9.0] * 8, [-9.0] * 8])
    idx = greedy_coreset(np.vstack([blob, outliers]).astype(np.float32), 20,
                         projection_dim=None)
    assert 400 in idx and 401 in idx


def test_whitener_equalises_comparable_feature_scales():
    """
    Features that genuinely vary, at wildly different units, must end up
    contributing equally. Without this, Euclidean distance is decided by whichever
    feature group happens to have the biggest numbers -- an accident of units, not
    a statement about defects.
    """
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.normal(0, 50.0, 400),      # e.g. a raw pixel statistic
                         rng.normal(0, 0.05, 400)]).astype(np.float32)  # e.g. a histogram bin
    w = Whitener.fit(x)
    z = (x - w.mean) / w.std
    assert abs(z[:, 0].std() - z[:, 1].std()) < 0.05


def test_whitener_damps_rather_than_amplifies_constant_features():
    """
    A dimension that never varies across enrolment carries no information about
    normal variation. Naive z-scoring divides by ~0 and lets that pure noise
    dominate every distance. The std floor must damp it instead.
    """
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.normal(0, 1.0, 400),
                         np.full(400, 7.0) + rng.normal(0, 1e-9, 400)]).astype(np.float32)
    w = Whitener.fit(x)
    z = (x - w.mean) / w.std
    assert z[:, 1].std() < z[:, 0].std(), "constant dimension was amplified, not damped"


def test_cascade_does_not_escalate_passing_parts(enrolled):
    class Boom:
        def explain(self, *a, **k):
            raise AssertionError("Tier 2 must never run on a PASS")
        def close(self): pass

    c = InspectionCascade(enrolled, Boom(), CascadePolicy())
    rng = np.random.default_rng(7)
    for i in range(5):
        rec = c.inspect(_part(rng), part_id=f"p{i}")
        if rec.result.verdict == "PASS":
            assert not rec.escalated


def test_drift_is_detected_on_passing_parts():
    """Drift must be visible before parts start failing -- that is the whole point."""
    base = [InspectionEvent(1e9 + i, "PASS", 0.30 + 0.004 * i) for i in range(60)]
    stats = compute_stats(base)
    assert stats.drift_detected
    assert stats.failed == 0, "drift must be detectable while yield is still 100%"


def test_no_drift_alarm_on_stable_line():
    rng = np.random.default_rng(3)
    stable = [InspectionEvent(1e9 + i, "PASS", 0.30 + rng.normal(0, 0.004)) for i in range(60)]
    assert not compute_stats(stable).drift_detected
