#!/usr/bin/env python3
"""
Tier 0 -- the gate. Runs on the Arduino UNO Q's Debian side (Dragonwing QRB2210).

Its whole job is to answer "is there a part in front of the camera right now?"
and, when the answer flips to yes, capture one sharp frame and ship it to the PC.

This is not a small thing to get right and it is not a small saving. A camera at
30 fps produces 108,000 frames an hour; a line at 40 parts/minute produces 2,400
parts. Screening every frame instead of every part would waste 97% of the NPU's
work on empty conveyor. The gate is what converts a video problem into a parts
problem.

It runs on the QRB2210's CPU using frame differencing and a sharpness check --
deliberately not a neural network. A model here would need its own enrolment,
its own failure modes, and its own explanation to the operator, to answer a
question that background subtraction answers correctly at 2 ms.

The STM32U585 MCU handles what Linux must not: the diverter solenoid timing.
See sketch/gate.ino.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import time

import cv2
import numpy as np

from .protocol import FrameHeader, DEFAULT_PORT, send_frame

log = logging.getLogger("unoq-gate")


class PartGate:
    """Rising-edge detector for 'a part entered the inspection window'."""

    def __init__(self, *, occupancy_threshold: float = 0.12,
                 sharpness_threshold: float = 45.0, settle_frames: int = 2,
                 cooldown_s: float = 0.25):
        self.occupancy_threshold = occupancy_threshold
        self.sharpness_threshold = sharpness_threshold
        self.settle_frames = settle_frames
        self.cooldown_s = cooldown_s
        self._background: np.ndarray | None = None
        self._present_for = 0
        self._armed = True
        self._last_fire = 0.0

    def update(self, frame: np.ndarray) -> tuple[bool, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 120))

        if self._background is None:
            self._background = small.astype(np.float32)
            return False, 0.0

        occupancy = float((np.abs(small.astype(np.float32) - self._background) > 18).mean())

        # Only adapt the background while the belt is empty, otherwise a part
        # that pauses gets absorbed into it and the next one is never detected.
        if occupancy < self.occupancy_threshold * 0.5:
            cv2.accumulateWeighted(small.astype(np.float32), self._background, 0.05)

        present = occupancy > self.occupancy_threshold
        self._present_for = self._present_for + 1 if present else 0
        if not present:
            self._armed = True
            return False, occupancy

        # Settle frames stop us grabbing a motion-blurred part mid-travel, and
        # the Laplacian check rejects the grab if it is blurred anyway.
        if (self._armed and self._present_for >= self.settle_frames
                and time.time() - self._last_fire > self.cooldown_s):
            if cv2.Laplacian(gray, cv2.CV_64F).var() >= self.sharpness_threshold:
                self._armed = False
                self._last_fire = time.time()
                return True, occupancy
        return False, occupancy


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="Snapdragon PC address on the factory LAN")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--node-id", default="unoq-01")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    gate = PartGate()
    sock = socket.create_connection((args.host, args.port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    log.info("Gate live. Streaming triggers to %s:%d", args.host, args.port)

    reader = sock.makefile("rb")
    count, t_gate = 0, []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            fire, occupancy = gate.update(frame)
            t_gate.append((time.perf_counter() - t0) * 1000)
            if not fire:
                continue

            count += 1
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if not ok:
                continue
            jpeg = buf.tobytes()
            send_frame(sock, FrameHeader(
                part_id=f"{args.node_id}-{count:06d}",
                captured_ms=time.monotonic() * 1000,
                trigger="part_present",
                jpeg_bytes=len(jpeg),
                gate_score=occupancy,
                node_id=args.node_id,
            ), jpeg)

            line = reader.readline()
            if line:
                v = json.loads(line)
                log.info("%s -> %s (severity %.2f)%s", v["part_id"], v["verdict"],
                         v["severity"], "  [REJECT]" if v["reject"] else "")
                if v["reject"]:
                    _fire_diverter()
            if count % 50 == 0:
                log.info("gate cost: %.2f ms/frame mean over %d frames",
                         float(np.mean(t_gate[-500:])), len(t_gate))
    except KeyboardInterrupt:
        log.info("Stopping. %d parts gated.", count)
    finally:
        cap.release()
        sock.close()


def _fire_diverter() -> None:
    """
    Hand the reject to the MCU.

    Deliberately a one-line message to the STM32 rather than a GPIO write from
    Python: Linux scheduling jitter is tens of milliseconds, and at 40 parts a
    minute that is enough to divert the wrong part. Determinism belongs on the
    Cortex-M33, which is exactly why the UNO Q has one.
    """
    try:
        from arduino.app_bricks.rpc import call  # provided by Arduino App Lab
        call("diverter.fire", duration_ms=120)
    except Exception:  # pragma: no cover - hardware only
        log.warning("Diverter RPC unavailable; reject not actuated")


if __name__ == "__main__":
    main()
