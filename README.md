# Feature-First ISR Demo

## Overview

This project demonstrates a feature-first ISR pipeline under constrained links.
It compares:

- baseline image streaming
- semantic streaming (tracks, events, and keyframes)

The application includes a link emulator, receiver UI, summary metrics, and RF silence simulation.

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended for real-time inference (CPU inference is ~1 fps)

## Install

```bash
python3 -m venv ~/venvs/ffisr
~/venvs/ffisr/bin/pip install -e .
```

### FIPS / air-gapped systems (Ubuntu)

The PyPI `opencv-python` wheel bundles its own OpenSSL which fails FIPS self-tests.
Use the system OpenCV instead and symlink it into the venv:

```bash
sudo apt install python3-opencv
pip uninstall -y opencv-python opencv-python-headless

SITEPACKAGES=$(~/venvs/ffisr/bin/python3 -c "import site; print(site.getsitepackages()[0])")
ln -s /usr/lib/python3/dist-packages/cv2 $SITEPACKAGES/cv2
```

## Model weights setup

Model weights are not stored in git (large binary files). Download them once into
the project root before starting the server:

```bash
# VisDrone-finetuned model (recommended for aerial footage)
curl -L -o yolov8s-visdrone.pt \
  "https://huggingface.co/mshamrai/yolov8s-visdrone/resolve/main/best.pt"

# Optional: standard COCO models (auto-downloaded by ultralytics on first use)
# yolov8s.pt, yolo11n.pt, etc.
```

## Run

```bash
~/venvs/ffisr/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`.

## Basic usage

1. Set link bandwidth, packet loss, and semantic share in the control panel.
2. Use 5 Mbps / 1 Mbps / 200 kbps presets for quick comparison.
3. Request keyframes for selected tracks.
4. Trigger RF silence to test blackout and reacquisition behavior.
5. Review metrics and summary text from the UI.

## API endpoints

- `GET /api/state`
- `POST /api/link_profile`
- `POST /api/request_keyframe`
- `POST /api/rf_silence`
- `GET /api/baseline_frame`
- `GET /api/keyframe`
- `GET /api/summary`

## Hardware integration references

- Integration docs: `app/integration/README.md`
- Example config: `integration.example.toml`
