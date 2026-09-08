#!/usr/bin/env python3
"""
Export a Tier-1 backbone from Qualcomm AI Hub, compiled for the Hexagon NPU.

Run this on an x64 Python environment (AI Hub's export tooling is not packaged
for ARM64 Python), then copy the resulting .onnx to the Snapdragon machine.

What this produces is a graph whose *intermediate* feature maps are outputs.
That is the non-obvious part: a stock classification export gives you 1000
logits, which are useless here -- we need the mid-level activations, because
those are what carry "this patch looks like a scratch" rather than "this image
is a goldfish".

Compilation targets `--target_runtime onnx` with QNN context binaries embedded,
so the file that lands on the PC needs no QAIRT-side conversion at first run.
"""

from __future__ import annotations

import argparse
import sys

DEVICES = {
    "x-elite": "Snapdragon X Elite CRD",
    "x2-elite": "Snapdragon X2 Elite CRD",
    "x-plus": "Snapdragon X Plus 8-Core CRD",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet18",
                    help="AI Hub model id used as the feature backbone")
    ap.add_argument("--device", choices=DEVICES, default="x2-elite")
    ap.add_argument("--out", default="models/")
    ap.add_argument("--input-size", type=int, default=320)
    args = ap.parse_args()

    try:
        import qai_hub as hub
    except ImportError:
        print("qai-hub is not installed.\n"
              "  pip install qai-hub qai-hub-models\n"
              "  qai-hub configure --api_token <token from aihub.qualcomm.com>\n"
              "Use an x64 Python environment for this step.", file=sys.stderr)
        return 1

    device = hub.Device(DEVICES[args.device])
    print(f"Target: {device.name}")

    # Compile for ONNX Runtime with QNN context binaries embedded, so the
    # artefact drops straight into onnxruntime-qnn on the laptop.
    compile_job = hub.submit_compile_job(
        model=args.model,
        device=device,
        input_specs={"image": (1, 3, args.input_size, args.input_size)},
        options="--target_runtime onnx --quantize_full_type w8a16",
    )
    print(f"Compile job:  {compile_job.url}")
    target_model = compile_job.get_target_model()

    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"Profile job:  {profile_job.url}")
    prof = profile_job.download_profile()

    ex = prof["execution_summary"]
    print("\nOn-device profile (real silicon, not an estimate):")
    print(f"  inference time      {ex['estimated_inference_time'] / 1000:.2f} ms")
    print(f"  peak memory         {ex['estimated_inference_peak_memory'] / 1e6:.1f} MB")
    print(f"  layers on NPU       {ex.get('compute_unit_counts', {}).get('NPU', 'n/a')}")
    print(f"  layers on CPU       {ex.get('compute_unit_counts', {}).get('CPU', 'n/a')}")
    print("\nA non-zero CPU layer count means part of the graph fell off the NPU. "
          "Check the profile page before trusting the latency.")

    target_model.download(args.out)
    print(f"\nDownloaded to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
