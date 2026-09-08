# Architecture

*NIRIKSHAK · Jash Pasad, IIT Gandhinagar · [github.com/jashpasad/NIRIKSHAK](https://github.com/jashpasad/NIRIKSHAK)*

## The one-sentence version

Four models of increasing cost, arranged so that expensive ones only see parts the cheap ones could not settle.

## Tier 0 — Gate (Arduino UNO Q, Dragonwing QRB2210)

**Question answered:** is there a part in front of the camera right now?

**Why it exists.** A camera at 30 fps produces 108,000 frames/hour; a line at 40 parts/min produces 2,400 parts. Without a gate, 97% of NPU work is spent on empty conveyor. The gate converts a video problem into a parts problem, and it is the single largest efficiency win in the system — larger than anything in Tier 1.

**Why it is not a neural network.** Frame differencing plus a Laplacian sharpness check answers this correctly in ~2 ms on the QRB2210's Cortex-A53 cores. A model here would need its own enrolment procedure, its own failure modes, and its own explanation to the operator — all to answer a question that background subtraction already answers. Complexity has to earn its place.

Three details that matter in practice:
- The background model only adapts while the belt is empty. Otherwise a part that pauses is absorbed into the background and the *next* part is never detected.
- Settle frames prevent grabbing a motion-blurred part mid-travel; the Laplacian check rejects the grab if it is blurred anyway.
- A rising-edge latch plus cooldown stops one part being reported as five.

**Why the MCU is separate.** The reject diverter fires from the STM32U585 running Zephyr, not from Python on Linux. A reject arriving 30 ms late has diverted the wrong part, and 30 ms of scheduling jitter is unremarkable for a Linux userspace process. Perception on the big cores, determinism on the Cortex-M33. This split is the reason a UNO Q is in the system rather than a plain USB camera.

## Tier 1 — Screen (Hexagon NPU)

**Question answered:** does this part contain a region unlike anything seen at enrolment, and where?

The pipeline, in order:

1. **Patch embedding.** A mid-level CNN backbone produces a grid of local descriptors. We concatenate an early feature map (texture, edges) with a deeper one (shape, part identity) and deliberately avoid the final layers — ImageNet's last block is optimised for "which of 1000 classes is this", which is precisely the wrong invariance. A scratch must *not* be invariant.

2. **Locally-aware pooling.** Each feature is averaged over a 3×3 neighbourhood. This buys tolerance to sub-millimetre conveyor jitter without a precision mechanical fixture — which is exactly the cost we are trying to remove from the cell.

3. **Whitening.** Per-dimension z-scoring against the enrolment set. Without it, distance is dominated by whichever feature group has the largest units. This was worth more than any other single change (AUROC 0.711 → 0.944).

4. **Coreset memory bank.** Greedy k-center repeatedly selects the vector furthest from everything already chosen. This *keeps* the rare-but-legitimate patches — a stamped logo, a chamfer, a colour transition — and discards the thousandth copy of flat background. Keeping outliers is the whole point: they are the good-part features a random subsample would drop, and dropping them is what produces false rejects.

5. **k-NN search as one GEMM.** Both sides are L2-normalised after whitening, so squared Euclidean distance is `2 − 2·cos` and the entire search is a single matrix product. For a 1,800-entry bank and a 1,521-patch query that is microseconds on the CPU, running while the NPU handles the next frame's backbone pass.

6. **Image score.** Mean of the top-3 most anomalous patches, not the single maximum. A max is one number from one patch and inherits that patch's noise; a real defect lights up a neighbourhood, so averaging the top few suppresses good-part outliers much more than genuine defects (AUROC 0.944 → 0.976).

7. **Calibration.** Leave-one-out over the enrolment set: score each image against a bank that has never seen it. That answers the only question relevant to threshold placement — how far from normal does a part that *is* normal actually land?

## Tier 2 — Explain (Hexagon NPU, Qwen3-VL-4B)

**Question answered:** what is this, in words an operator can act on?

Tier 1 says "something at (412, 260) is 3.1σ from normal." That is actionable for an engineer and useless for an operator.

Three constraints:

- **A crop, not the full frame.** A VLM asked to find a 4 mm scratch in a 12 MP image will confabulate. Shown the scratch, it describes it. We use the cheap model to aim the expensive one — that is what the Tier-1 heatmap is *for*.
- **Constrained JSON, not prose.** Free text cannot be logged, counted or trended, and an inspection record that cannot be trended is decoration.
- **Never the decision-maker.** Tier 1's calibrated distance decides PASS/REVIEW/FAIL. Tier 2 supplies vocabulary. Letting a generative model adjudicate a physical accept/reject would be indefensible on an audited line.

The escalation policy (`CascadePolicy`) is an explicit, inspectable rule set with a minimum-severity floor and a per-minute escalation budget. If the line goes badly wrong we would rather keep screening every part at 10 ms than fall behind narrating a flood of failures. **Throughput protection beats completeness.**

## Tier 3 — Reason (Hexagon NPU, Qwen3-4B)

**Question answered:** what happened this shift, and what does it say about the process?

Statistics are computed in numpy *before* any model sees anything — the model is given results, never asked to derive them. Drift detection compares the mean anomaly score of the first and last quarters of the shift, **restricted to passing parts**: if good parts are creeping toward the threshold, the process is moving before it produces scrap.

If the LLM is unavailable, a deterministic report is emitted anyway. A line that stops producing its shift record because a model failed to load is worse than one producing a plainer record.

## Data flow and failure behaviour

| Component fails | System behaviour |
|---|---|
| GenieX absent | Tiers 2/3 disabled; Tier 0/1 unaffected; deterministic report still written |
| QNN EP rejects the graph | Falls back to DirectML then CPU, and **says so** in telemetry rather than silently running 40× slower |
| UNO Q offline | PC accepts manually triggered or folder-batch input |
| Network absent | No effect — there is no network dependency anywhere in the design |
