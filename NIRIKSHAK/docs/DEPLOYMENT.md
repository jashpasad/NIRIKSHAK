# Deployment

*NIRIKSHAK · Jash Pasad, IIT Gandhinagar · [github.com/jashpasad/NIRIKSHAK](https://github.com/jashpasad/NIRIKSHAK)*

## The setup mistake almost everyone makes

Qualcomm's tooling needs **two** Python environments on a Snapdragon PC, and mixing them is the most common failure:

| Purpose | Python | Why |
|---|---|---|
| Model export / AI Hub compile / profiling | **x64 (AMD64)** | `qai-hub-models` and its torch dependency are not packaged for ARM64 Python |
| On-device inference | **ARM64** | `onnxruntime-qnn` and `geniex` load native ARM64 libraries |

Export on x64, copy the artefact, run on ARM64. Do not try to do both in one environment.

## Snapdragon PC

```powershell
# ARM64 Python 3.12 — inference environment
py -3.12-arm64 -m venv .venv-arm && .venv-arm\Scripts\activate
pip install -r requirements-snapdragon.txt
```

```powershell
# x64 Python 3.10 — export environment
py -3.10 -m venv .venv-x64 && .venv-x64\Scripts\activate
pip install qai-hub qai-hub-models
qai-hub configure --api_token <token from aihub.qualcomm.com>
python scripts/fetch_backbone.py --device x2-elite --out models/
```

Verify the NPU is actually being used:

```python
from nirikshak.runtime.backend import OnnxBackend
b = OnnxBackend("models/backbone.onnx", context_cache="models/ctx.bin")
print(b.compute_unit)   # want: Hexagon NPU [QNNExecutionProvider] (accelerated)
```

If it prints `Oryon CPU`, the QNN EP rejected the graph and silently fell back. The backend logs a warning when this happens — **it never pretends**. A cell that quietly drops from 10 ms to 400 ms per part is a broken cell.

### QNN provider options we set, and why

| Option | Value | Reason |
|---|---|---|
| `backend_path` | `QnnHtp.dll` | selects the HTP (NPU), not QNN's CPU reference backend |
| `htp_performance_mode` | `burst` | right for a line-rate loop; use `sustained_high_performance` for the Tier-3 batch report |
| `qnn_context_cache_enable` | `1` | ~20 s cold graph finalisation → sub-second warm start |
| `enable_htp_fp16_precision` | `1` | fp16 accumulation where the graph allows it |

Context caching matters more than it looks: an operator power-cycles the cell every shift, and twenty seconds of staring at a splash screen every morning is how a deployment loses its welcome.

## Arduino UNO Q

Debian side (Tier-0 gate):

```bash
sudo apt install python3-opencv python3-numpy
python3 -m nirikshak.edge.uno_q_agent --host <pc-ip> --camera 0
```

MCU side: flash `nirikshak/edge/sketch/gate.ino` via Arduino App Lab. Pinout:

| Pin | Function |
|---|---|
| D9 | diverter solenoid driver |
| D6 | green stack lamp (pass) |
| D5 | red stack lamp (review / fail) |
| D2 | optional hardware part-present sensor (interrupt, `INPUT_PULLUP`) |

The link is newline-delimited JSON plus JPEG over plain TCP. No broker, no MQTT, no cloud tenant — this runs on a factory LAN that frequently has no route to the internet, and every added dependency is one an electrician has to debug at 2 a.m.

## Bill of materials

| Item | ₹ |
|---|---|
| HP Snapdragon PC (X-series) | ~95,000 |
| Arduino UNO Q (4 GB) | ~6,000 |
| USB camera (1080p, fixed focus) | ~3,000 |
| LED ring light + mount | ~4,000 |
| Diverter solenoid + 24 V supply | ~3,000 |
| **Total** | **~1,11,000** |

Against ₹15–40 lakh for a conventional machine-vision cell. Software cost is zero and marginal inference cost is electricity.

## Commissioning a new part

1. Fixture the part and set the lighting. Do this once and do not move it — the system models *this* view.
2. Run 20+ known-good parts through: `nirikshak enrol ./good -o part.npz`
3. Check the printed `pass_below` / `fail_above` and the safe sigma window from `scripts/evaluate.py`.
4. Run one shift with the diverter disabled, in advisory mode. Review the REVIEW bin with the supervisor.
5. Retune `pass_sigma` against that shift's data, then enable the diverter.

Step 4 is not optional. Shipping thresholds derived from 20 images straight into a live reject loop is how an operator loses trust in the machine in a single morning — and an operator who switches it off has a system that detects nothing.

## Re-enrolling

Re-enrol when: tooling changes, material lot changes, the lighting or camera is disturbed, or Tier-3 drift alarms persist across shifts with no assignable process cause. Enrolment takes ~5 s, so re-enrolling is cheap; treat it as routine maintenance rather than an event.
