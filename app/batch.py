from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.codec import bbox_delta, quantize_confidence, quantize_velocity, serialize_json, speed
from app.transport import LinkEmulator

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _discover_sequences(dataset_dir: str) -> list[Path]:
    root = Path(dataset_dir)
    if not root.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    sequences: list[Path] = []
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir():
            images = [f for f in candidate.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
            if images:
                sequences.append(candidate)

    if not sequences:
        images = [f for f in root.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
        if images:
            sequences.append(root)

    return sequences


def _load_frames(sequence_dir: Path, target_w: int, target_h: int) -> list[np.ndarray]:
    files = sorted(f for f in sequence_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    frames: list[np.ndarray] = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[1] != target_w or img.shape[0] != target_h:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(img)
    return frames


def _encode_jpeg(frame: np.ndarray, quality: int = 60) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def _run_sequence(
    seq_dir: Path,
    model: Any,
    tracker: str,
    conf: float,
    link_kbps: int,
    packet_loss: float,
    semantic_ratio: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    frames = _load_frames(seq_dir, width, height)
    if not frames:
        return {"sequence": seq_dir.name, "error": "no readable frames"}

    link = LinkEmulator(link_kbps=link_kbps, packet_loss=packet_loss, semantic_ratio=semantic_ratio)

    track_ids_seen: set[int] = set()
    track_prev_bbox: dict[int, list[int]] = {}
    track_last_centers: dict[int, tuple[float, float, float]] = {}

    semantic_delivered = 0
    baseline_delivered = 0
    semantic_attempted_bytes = 0
    baseline_attempted_bytes = 0
    semantic_attempted = 0
    baseline_attempted = 0

    sim_ts = 0.0
    dt = 1.0 / 8.0

    for frame in frames:
        sim_ts += dt

        jpeg = _encode_jpeg(frame)
        baseline_attempted += 1
        baseline_attempted_bytes += len(jpeg)
        if link.send("baseline", len(jpeg)):
            baseline_delivered += len(jpeg)

        try:
            results = model.track(
                source=frame,
                conf=conf,
                tracker=tracker,
                persist=True,
                verbose=False,
            )
        except Exception:
            continue

        if not results:
            continue

        first = results[0]
        names_raw = getattr(first, "names", {})
        names: dict[int, str] = {int(k): str(v) for k, v in names_raw.items()} if isinstance(names_raw, dict) else {}
        boxes = getattr(first, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            box_id = getattr(box, "id", None)
            if box_id is None:
                continue
            track_id = int(float(box_id.item()))
            track_ids_seen.add(track_id)

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            bbox = [x1, y1, x2 - x1, y2 - y1]

            delta, _ = bbox_delta(bbox, track_prev_bbox.get(track_id))
            cls_idx = int(float(box.cls.item())) if getattr(box, "cls", None) is not None else -1
            cls_name = names.get(cls_idx, f"class_{cls_idx}")
            conf_raw = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0
            conf_q = quantize_confidence(max(0.0, min(1.0, conf_raw)))

            cx = x1 + bbox[2] / 2.0
            cy = y1 + bbox[3] / 2.0
            prev_center = track_last_centers.get(track_id)
            velocity_raw = 0.0
            if prev_center is not None:
                dt_c = max(1e-3, sim_ts - prev_center[2])
                velocity_raw = speed(cx - prev_center[0], cy - prev_center[1]) / dt_c
            track_last_centers[track_id] = (cx, cy, sim_ts)
            velocity = quantize_velocity(velocity_raw)

            message = {
                "type": "TRACK_UPDATE",
                "track_id": track_id,
                "class": cls_name,
                "confidence": conf_q,
                "velocity": velocity,
                "timestamp": round(sim_ts, 2),
                "bbox": bbox if delta is None else None,
                "bbox_delta": delta,
            }
            packed = serialize_json(message)
            semantic_attempted += 1
            semantic_attempted_bytes += len(packed)
            if link.send("semantic", len(packed)):
                semantic_delivered += len(packed)
                track_prev_bbox[track_id] = bbox

    elapsed = max(1.0, sim_ts)
    sem_kbps = (semantic_delivered * 8.0 / elapsed) / 1000.0
    base_kbps = (baseline_delivered * 8.0 / elapsed) / 1000.0
    sem_attempted_kbps = (semantic_attempted_bytes * 8.0 / elapsed) / 1000.0
    base_attempted_kbps = (baseline_attempted_bytes * 8.0 / elapsed) / 1000.0
    bw_savings = 0.0
    if baseline_attempted_bytes > 0:
        bw_savings = max(0.0, (1.0 - semantic_attempted_bytes / float(baseline_attempted_bytes)) * 100.0)

    sem_drop_pct = 0.0
    if semantic_attempted > 0:
        sem_drop_pct = ((semantic_attempted - (semantic_delivered if semantic_delivered == 0 else 0)) / semantic_attempted) * 100.0

    stats = link.stats()
    sem_stats = stats.get("semantic", {})
    base_stats = stats.get("baseline", {})
    sem_drop_pct = 0.0
    if sem_stats.get("attempted_packets", 0) > 0:
        sem_drop_pct = sem_stats["dropped_packets"] / sem_stats["attempted_packets"] * 100.0
    base_drop_pct = 0.0
    if base_stats.get("attempted_packets", 0) > 0:
        base_drop_pct = base_stats["dropped_packets"] / base_stats["attempted_packets"] * 100.0

    return {
        "sequence": seq_dir.name,
        "frames": len(frames),
        "unique_tracks": len(track_ids_seen),
        "semantic_kbps": round(sem_kbps, 2),
        "baseline_kbps": round(base_kbps, 2),
        "sem_attempted_kbps": round(sem_attempted_kbps, 2),
        "base_attempted_kbps": round(base_attempted_kbps, 2),
        "bandwidth_savings_pct": round(bw_savings, 1),
        "semantic_drop_pct": round(sem_drop_pct, 1),
        "baseline_drop_pct": round(base_drop_pct, 1),
        "power_proxy_semantic_mbits": round((semantic_attempted_bytes * 8.0) / 1_000_000.0, 3),
        "power_proxy_baseline_mbits": round((baseline_attempted_bytes * 8.0) / 1_000_000.0, 3),
        "error": None,
    }


def run_batch(
    dataset_dir: str,
    model_path: str = "yolo11n.pt",
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
    link_kbps: int = 1000,
    packet_loss: float = 0.02,
    semantic_ratio: float = 0.65,
    max_sequences: int = 5,
    width: int = 960,
    height: int = 540,
) -> dict[str, Any]:
    try:
        ultralytics = importlib.import_module("ultralytics")
    except Exception:
        raise RuntimeError("ultralytics is not installed; run: pip install ultralytics")

    try:
        model = ultralytics.YOLO(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_path}': {exc}")

    sequences = _discover_sequences(dataset_dir)
    if not sequences:
        raise ValueError(f"No image sequences found under: {dataset_dir}")

    sequences = sequences[:max_sequences]
    results: list[dict[str, Any]] = []

    for seq in sequences:
        result = _run_sequence(
            seq_dir=seq,
            model=model,
            tracker=tracker,
            conf=conf,
            link_kbps=link_kbps,
            packet_loss=packet_loss,
            semantic_ratio=semantic_ratio,
            width=width,
            height=height,
        )
        results.append(result)

    valid = [r for r in results if r.get("error") is None]
    summary: dict[str, Any] = {}
    if valid:
        summary = {
            "sequences_evaluated": len(valid),
            "total_frames": sum(r["frames"] for r in valid),
            "total_unique_tracks": sum(r["unique_tracks"] for r in valid),
            "avg_semantic_kbps": round(sum(r["semantic_kbps"] for r in valid) / len(valid), 2),
            "avg_baseline_kbps": round(sum(r["baseline_kbps"] for r in valid) / len(valid), 2),
            "avg_bandwidth_savings_pct": round(sum(r["bandwidth_savings_pct"] for r in valid) / len(valid), 1),
            "avg_semantic_drop_pct": round(sum(r["semantic_drop_pct"] for r in valid) / len(valid), 1),
            "avg_baseline_drop_pct": round(sum(r["baseline_drop_pct"] for r in valid) / len(valid), 1),
            "total_power_proxy_semantic_mbits": round(sum(r["power_proxy_semantic_mbits"] for r in valid), 3),
            "total_power_proxy_baseline_mbits": round(sum(r["power_proxy_baseline_mbits"] for r in valid), 3),
        }
        if summary["total_power_proxy_baseline_mbits"] > 0:
            summary["power_savings_pct"] = round(
                (1.0 - summary["total_power_proxy_semantic_mbits"] / summary["total_power_proxy_baseline_mbits"]) * 100.0,
                1,
            )
        else:
            summary["power_savings_pct"] = 0.0

    return {
        "config": {
            "model_path": model_path,
            "tracker": tracker,
            "conf": conf,
            "link_kbps": link_kbps,
            "packet_loss": packet_loss,
            "semantic_ratio": semantic_ratio,
        },
        "results": results,
        "summary": summary,
    }
