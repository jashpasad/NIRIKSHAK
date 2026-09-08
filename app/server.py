"""
Local inspection server: TCP listener for the UNO Q gate + a browser UI.

Binds to 127.0.0.1 by default. There is no authentication because there is no
remote surface: the operator UI is on the machine doing the inspecting, and the
edge link lives on the factory LAN. Adding a login screen to a machine bolted to
a conveyor would be security theatre with a real usability cost.
"""

from __future__ import annotations

import base64
import json
import logging
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from nirikshak import AnomalyDetector, ReferencePatchEmbedder, InspectionCascade, CascadePolicy
from nirikshak.edge.protocol import DEFAULT_PORT, Verdict, recv_frame

log = logging.getLogger("nirikshak.server")

STATE: dict = {"recent": [], "counts": {"PASS": 0, "REVIEW": 0, "FAIL": 0}, "cascade": None}
_LOCK = threading.Lock()


def _record(part_id: str, image: np.ndarray, rec) -> None:
    hm = cv2.applyColorMap((np.clip(rec.result.heatmap, 0, 1) * 255).astype(np.uint8),
                           cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.55, hm, 0.45, 0)
    thumb = cv2.resize(overlay, (220, 220))
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    with _LOCK:
        STATE["counts"][rec.result.verdict] += 1
        STATE["recent"].insert(0, {
            "part_id": part_id,
            "verdict": rec.result.verdict,
            "score": round(rec.result.score, 4),
            "latency_ms": round(rec.total_ms, 1),
            "escalated": rec.escalated,
            "defect": rec.explanation.defect_type if rec.explanation else None,
            "description": rec.explanation.description if rec.explanation else None,
            "thumb": base64.b64encode(buf).decode() if ok else "",
            "t": time.time(),
        })
        del STATE["recent"][60:]


def _edge_listener(cascade: InspectionCascade, port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(4)
    log.info("Edge listener on :%d", port)
    while True:
        conn, addr = srv.accept()
        log.info("Edge node connected from %s", addr[0])
        threading.Thread(target=_handle_edge, args=(conn, cascade), daemon=True).start()


def _handle_edge(conn: socket.socket, cascade: InspectionCascade) -> None:
    try:
        while True:
            header, jpeg = recv_frame(conn)
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            rec = cascade.inspect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                 part_id=header.part_id, compute_unit="Hexagon NPU")
            _record(header.part_id, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rec)
            conn.sendall(Verdict(header.part_id, rec.result.verdict,
                                 rec.result.severity,
                                 rec.result.verdict == "FAIL").encode())
    except (ConnectionError, OSError) as exc:
        log.info("Edge node disconnected: %s", exc)
    finally:
        conn.close()


INDEX = """<!doctype html><meta charset=utf-8><title>NIRIKSHAK</title>
<style>
:root{--bg:#111417;--card:#1a1e22;--fg:#e8eaed;--mut:#8b949e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:18px 24px;border-bottom:1px solid #262b31;display:flex;
align-items:baseline;gap:16px}h1{margin:0;font-size:19px;letter-spacing:.5px}
.sub{color:var(--mut);font-size:12px}
.kpis{display:flex;gap:14px;padding:20px 24px}
.kpi{background:var(--card);border-radius:10px;padding:14px 20px;min-width:120px}
.kpi b{display:block;font-size:30px;font-weight:600;line-height:1.1}
.kpi span{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.7px}
.PASS{color:#3fb950}.REVIEW{color:#d29922}.FAIL{color:#f85149}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
gap:14px;padding:0 24px 32px}
.c{background:var(--card);border-radius:10px;overflow:hidden}
.c img{width:100%;display:block}
.m{padding:10px 12px;font-size:12px}
.m .v{font-weight:600}.m .d{color:var(--mut);margin-top:4px}
.empty{padding:60px 24px;color:var(--mut)}
</style>
<header><h1>NIRIKSHAK</h1><span class=sub id=unit>on-device · no data leaves this machine</span></header>
<div class=kpis id=kpis></div><div class=grid id=grid></div>
<script>
async function tick(){
 const s=await (await fetch('/api/state')).json();
 const t=s.counts.PASS+s.counts.REVIEW+s.counts.FAIL;
 document.getElementById('kpis').innerHTML=
  `<div class=kpi><b>${t}</b><span>inspected</span></div>
   <div class=kpi><b class=PASS>${s.counts.PASS}</b><span>pass</span></div>
   <div class=kpi><b class=REVIEW>${s.counts.REVIEW}</b><span>review</span></div>
   <div class=kpi><b class=FAIL>${s.counts.FAIL}</b><span>fail</span></div>
   <div class=kpi><b>${t?((100*s.counts.PASS/t).toFixed(1)):'--'}%</b><span>yield</span></div>`;
 const g=document.getElementById('grid');
 g.innerHTML = s.recent.length ? s.recent.map(r=>
  `<div class=c><img src="data:image/jpeg;base64,${r.thumb}">
   <div class=m><span class="v ${r.verdict}">${r.verdict}</span>
   &middot; ${r.score} &middot; ${r.latency_ms} ms
   ${r.description?`<div class=d>${r.defect}: ${r.description}</div>`:''}</div></div>`).join('')
  : '<div class=empty>Waiting for parts from the edge node&hellip;</div>';
}
tick(); setInterval(tick,900);
</script>"""


def serve(recipe: str | Path, backbone: str | None = None,
          host: str = "127.0.0.1", port: int = 8000,
          edge_port: int = DEFAULT_PORT) -> None:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    import uvicorn

    embedder = ReferencePatchEmbedder()
    if backbone:
        from nirikshak.runtime.backend import OnnxBackend
        from nirikshak.tier1_screen.embedder import OnnxPatchEmbedder
        b = OnnxBackend(backbone)
        log.info("compute: %s", b.compute_unit)
        embedder = OnnxPatchEmbedder(b)

    detector = AnomalyDetector.load(recipe, embedder)
    cascade = InspectionCascade(detector, None, CascadePolicy())
    STATE["cascade"] = cascade
    threading.Thread(target=_edge_listener, args=(cascade, edge_port), daemon=True).start()

    api = FastAPI(title="NIRIKSHAK")

    @api.get("/", response_class=HTMLResponse)
    def index():
        return INDEX

    @api.get("/api/state")
    def state():
        with _LOCK:
            return {"counts": dict(STATE["counts"]), "recent": list(STATE["recent"][:24]),
                    "telemetry": cascade.telemetry.summary()}

    uvicorn.run(api, host=host, port=port, log_level="warning")
