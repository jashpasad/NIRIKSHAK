"""
Line protocol between the UNO Q edge node and the Snapdragon PC.

Newline-delimited JSON over a plain TCP socket. No broker, no MQTT, no cloud
tenant. That is a deliberate choice: this link lives entirely inside a factory
LAN that frequently has no route to the internet at all, and every dependency
added here is a dependency an electrician has to debug at 2 a.m.

Frames are sent as a JSON header line followed by raw JPEG bytes, so the header
stays greppable with `nc` while the payload stays compact.
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass, asdict

MAGIC = b"NRK1"
DEFAULT_PORT = 9707


@dataclass
class FrameHeader:
    part_id: str
    captured_ms: float          # UNO Q monotonic clock at shutter
    trigger: str                # "part_present" | "manual" | "interval"
    jpeg_bytes: int
    gate_score: float = 0.0     # Tier-0 confidence that a part is in frame
    node_id: str = "unoq-01"


def send_frame(sock: socket.socket, header: FrameHeader, jpeg: bytes) -> None:
    blob = json.dumps(asdict(header)).encode("utf-8")
    sock.sendall(MAGIC + struct.pack("!I", len(blob)) + blob + jpeg)


def recv_frame(sock: socket.socket) -> tuple[FrameHeader, bytes]:
    if _recv_exact(sock, 4) != MAGIC:
        raise ConnectionError("bad magic -- stream desynchronised")
    (hlen,) = struct.unpack("!I", _recv_exact(sock, 4))
    header = FrameHeader(**json.loads(_recv_exact(sock, hlen)))
    return header, _recv_exact(sock, header.jpeg_bytes)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-frame")
        buf += chunk
    return bytes(buf)


@dataclass
class Verdict:
    """PC -> UNO Q. The MCU acts on this within one conveyor index."""

    part_id: str
    verdict: str                # PASS | REVIEW | FAIL
    severity: float
    reject: bool                # fire the diverter

    def encode(self) -> bytes:
        return (json.dumps(asdict(self)) + "\n").encode("utf-8")
