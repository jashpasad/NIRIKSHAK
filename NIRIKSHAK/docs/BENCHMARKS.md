# Benchmarks

*NIRIKSHAK · Jash Pasad, IIT Gandhinagar · [github.com/jashpasad/NIRIKSHAK](https://github.com/jashpasad/NIRIKSHAK)*

Everything here is reproducible from this repository. Commands are given for each table.

## Honesty statement

The headline synthetic result is AUROC 1.000. **We generated that data, so we do not get to be impressed by it.** It demonstrates that the pipeline is correct and reproducible on any machine. It is not a claim about field accuracy.

Two things make it non-trivial rather than meaningless:

1. The generator injects the nuisance variation that normally defeats this class of method — randomised lighting gradient, ±4 px translation, ±1.4° rotation, Gaussian sensor noise, and a brushed finish whose parallel streaks are visually confusable with scratches. Without those, any detector separates a defect from a pixel-identical template.
2. The evaluator reports escapes and false rejects separately and never collapses REVIEW into either.

The external reference that carries weight is **MVTec AD**, where published PatchCore-class methods reach ~0.99 image AUROC. `scripts/evaluate.py` accepts MVTec's directory layout unchanged:

```bash
python scripts/evaluate.py --data /path/to/mvtec/screw
```

Running that is the first task of the next round, alongside a real camera trial.

## Detection

```bash
python scripts/make_demo_data.py
python scripts/evaluate.py --data data/demo
```

24 enrolment samples, 40 held-out good parts, 40 defective across 5 types.

| Defect type | n | AUROC | caught | escaped | localised |
|---|---|---|---|---|---|
| scratch | 8 | 1.000 | 8 | 0 | 8 |
| dent | 8 | 1.000 | 8 | 0 | 8 |
| chip | 8 | 1.000 | 8 | 0 | 8 |
| contamination | 8 | 1.000 | 8 | 0 | 8 |
| discolouration | 8 | 1.000 | 8 | 0 | 8 |
| **overall** | **40** | **1.000** | **40** | **0** | **40** |

- Escape rate **0%** — no defective part marked PASS
- False reject rate **0/40** — no good part marked FAIL
- Separation gap: good max `0.5532` < defect min `0.5872`
- Safe `pass_sigma` window `[0.78, 2.06]`; shipped value `1.50` (the centre)

The safe window matters more than the AUROC. It says how much margin the separation has before a threshold choice starts costing something, and it is why we ship the middle of the window rather than a value tuned to the edge.

## Ablation — how we got here

Each row is the configuration change alone, all else fixed.

| Configuration | AUROC | escape rate | enrolment |
|---|---|---|---|
| Baseline: 36-dim descriptor, no whitening, non-overlapping, single bg kernel | 0.711 | 77.5% | 6.9 s |
| + per-dimension whitening | 0.944 | 57.5% | 8.4 s |
| + top-3 score pooling | 0.976 | — | — |
| + overlapping patches (stride 8) | 0.999 | — | — |
| + dual background kernels (65, 161) | **1.000** | 7.5% | 92.7 s |
| + fixed LOO coreset-size bug | 1.000 | **0%** | 92.7 s |
| + origin-masked LOO calibration | 1.000 | 0% | **5.3 s** |

Per-defect recall through the two structural changes:

| | scratch | dent | chip | contamination | discolouration |
|---|---|---|---|---|---|
| non-overlapping, 1 bg kernel | 38% | 38% | 75% | 88% | 100% |
| overlapping, 1 bg kernel | 100% | 88% | 100% | 100% | 100% |
| overlapping, 2 bg kernels | 100% | 100% | 100% | 100% | 100% |

Two lessons worth carrying:

- **Overlapping patches fixed scratches.** A thin defect crossing a tile boundary is diluted into two patches and caught in neither.
- **The second background kernel fixed dents.** A 40 px dent inside a 65 px background window is absorbed into its own background and vanishes. A single local-background width can only see defects smaller than itself.

And one that is uncomfortable: at the point the escape rate was 57.5%, **AUROC was already 0.944**. The detector was fine and the thresholds were broken. A separability metric cannot see a calibration bug — which is why this repository reports escapes and false rejects as first-class numbers.

## Latency and cascade economics

```bash
python scripts/benchmark.py --iters 200
```

Tier-1 measured, reference descriptor, x86-64 CPU (the pessimistic path — no NPU, no CNN backbone):

| | ms |
|---|---|
| mean | 52.0 |
| p50 | 52.1 |
| p95 | 55.0 |
| p99 | 59.1 |

Enrolment: 5.3 s for 24 samples → 1,825 coreset entries from 36,504 descriptors (5.0%). Recipe on disk: 320 KB.

Cascade economics, measured Tier-1 against a 400 ms Tier-2:

| Line yield | Escalation rate | Effective ms/part | Parts/min | vs VLM-on-everything |
|---|---|---|---|---|
| 90% | 10.9% | 95.6 | 627 | 4.2× |
| 95% | 6.0% | 75.8 | 791 | 5.3× |
| 97% | 4.0% | 67.9 | 883 | 5.9× |
| 99% | 2.0% | 60.0 | 1000 | 6.7× |

**The gating gain grows as the line gets better**, which is the right incentive: a well-run line pays almost nothing for the explanation tier.

## Target device figures (modelled, not yet measured by us)

Clearly separated because they are not our measurements. Sources: Qualcomm AI Hub published profiles for Snapdragon X2 Elite.

| Stage | Model | Target | Status |
|---|---|---|---|
| Tier 0 gate | frame diff | ~2 ms | measured, QRB2210 class CPU |
| Tier 1 screen | CNN backbone, W8A16 | ~10 ms | modelled from AI Hub profiles |
| Tier 2 explain | Qwen3-VL-4B | ~400 ms | modelled |
| Tier 3 reason | Qwen3-4B, ~700 tok | ~25 s | modelled |

`scripts/fetch_backbone.py` submits a real profile job to Qualcomm AI Hub and prints measured on-device inference time, peak memory, and the NPU/CPU layer split. **A non-zero CPU layer count means part of the graph fell off the NPU and the latency should not be trusted** — the script says so explicitly.

Replacing every modelled number above with a profiled one is the first task of the next round.
