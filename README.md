# Feature-First ISR Demo

## Overview

This project demonstrates a feature-first ISR pipeline under constrained links.
It compares:

- baseline image streaming
- semantic streaming (tracks, events, and keyframes)

The application includes a link emulator, receiver UI, summary metrics, and RF silence simulation.

## Requirements

- Python 3.10+

## Install

```bash
~/venvs/ffisr/bin/pip install -e .
```

## Run

```bash
~/venvs/ffisr/bin/uvicorn app.main:app --reload
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
