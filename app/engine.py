from __future__ import annotations

import importlib
from pathlib import Path
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.codec import bbox_delta, quantize_confidence, quantize_velocity, reduced_embedding, serialize_json, speed
from app.transport import LinkEmulator

# COCO class index → clean aerial label.
# Only classes that are meaningful in top-down drone imagery are kept;
# everything else (kitchen items, animals, sports gear, etc.) is discarded.
_COCO_AERIAL_FILTER: dict[int, str] = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    8:  "boat",
}

# VisDrone class names are already aerial-appropriate; no remapping needed.
_VISDRONE_CLASSES = {
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
}


@dataclass
class SimObject:
    gt_id: int
    cls: str
    x: float
    y: float
    w: float
    h: float
    vx: float
    vy: float
    born_ts: float
    loiter_start_ts: float | None = None


class FeatureFirstEngine:
    def __init__(self, width: int = 960, height: int = 540, fps: int = 8) -> None:
        self.width = width
        self.height = height
        self.fps = fps

        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

        self.link = LinkEmulator(link_kbps=5000, packet_loss=0.02, semantic_ratio=0.55)

        self.start_monotonic = time.monotonic()
        self.tick_count = 0

        self.objects: list[SimObject] = []
        self.next_gt_id = 1
        self.next_track_id = 100

        self.gt_to_track: dict[int, int] = {}
        self.track_prev_bbox: dict[int, list[int]] = {}
        self.track_last_sent_ts: dict[int, float] = {}

        self.track_view: dict[int, dict[str, Any]] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=80)

        self.latest_baseline_frame = self._blank_frame()
        self.latest_baseline_frame_id = 0
        self.latest_keyframe = self._blank_frame((30, 30, 30))
        self.latest_keyframe_id = 0

        self.pending_keyframe_track: int | None = None
        self.force_keyframe = False
        self.last_auto_keyframe_ts = -999.0
        self.last_periodic_keyframe_ts = -999.0

        self.enter_ts_by_gt: dict[int, float] = {}
        self.detect_ts_by_gt: dict[int, float] = {}
        self.exits_total = 0
        self.missed_total = 0
        self.id_switches = 0
        self.track_assignments = 0

        self.delivered_samples: list[tuple[float, str, int]] = []

        self.rf_silence_active = False
        self.rf_silence_started_monotonic = 0.0
        self.rf_total_silence_s = 0.0
        self.rf_last_silence_s = 0.0
        self.rf_reconnect_pending = False
        self.rf_reconnect_started_monotonic = 0.0
        self.rf_reconnect_latency_ms = 0.0
        self.rf_reconnect_catchup_bytes = 0
        self.rf_buffered_tracks: dict[int, dict[str, Any]] = {}
        self.rf_buffered_events: deque[dict[str, Any]] = deque(maxlen=160)
        self.rf_reconnect_queue: deque[tuple[str, dict[str, Any]]] = deque()

        self.video_source_mode = "synthetic"
        self.video_source_path: str | None = None
        self.video_loop = True
        self.video_capture: cv2.VideoCapture | None = None
        self.video_sequence_files: list[str] = []
        self.video_sequence_index = 0
        self.video_frame_index = 0
        self.video_last_error: str | None = None
        self.video_last_frame: np.ndarray | None = None

        self.tracking_enabled = True
        self.tracking_model_path = "yolov8s.pt"
        self.tracking_conf = 0.25
        self.tracking_tracker = "app/bytetrack_aerial.yaml"
        self.tracking_model: Any | None = None
        self.tracking_last_error: str | None = None
        self._coco_filter_active = self._is_coco_model(self.tracking_model_path)
        self.tracking_track_centers: dict[int, tuple[float, float, float]] = {}
        self.preprocess_mode = "none"
        self._model_loading = False
        self._pending_model_path: str | None = None
        self._last_inference_frame: np.ndarray | None = None
        self._last_tick_frame: np.ndarray | None = None
        self._last_tick_sim_ts: float = 0.0
        self._last_tick_now: float = 0.0

    def _blank_frame(self, color: tuple[int, int, int] = (18, 18, 18)) -> bytes:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = color
        _, encoded = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        )
        return encoded.tobytes()

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            self.running = True
            self.start_monotonic = time.monotonic()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        with self.lock:
            self.running = False
            thread = self.thread
        if thread is not None:
            thread.join(timeout=1.0)
        with self.lock:
            self._close_video_capture()

    def set_link_profile(self, link_kbps: int, packet_loss: float, semantic_ratio: float) -> None:
        with self.lock:
            self.link.set_profile(link_kbps=link_kbps, packet_loss=packet_loss, semantic_ratio=semantic_ratio)

    def request_keyframe(self, track_id: int | None = None) -> None:
        with self.lock:
            self.pending_keyframe_track = track_id
            self.force_keyframe = True

    def trigger_rf_silence(self, duration_s: float) -> None:
        with self.lock:
            self.link.start_rf_silence(duration_s)

    def set_tracking_config(
        self,
        enabled: bool = True,
        model_path: str = "yolov8s.pt",
        conf: float = 0.35,
        tracker: str = "bytetrack.yaml",
        preprocess_mode: str = "visible",
    ) -> dict[str, Any]:
        with self.lock:
            model_path = model_path.strip() or "yolov8s.pt"
            tracker = tracker.strip() or "bytetrack.yaml"
            preprocess_mode = preprocess_mode.strip().lower() or "visible"

            if model_path != self.tracking_model_path:
                self.tracking_model = None
                self._model_loading = False
            if tracker != self.tracking_tracker:
                self.tracking_track_centers.clear()

            self.tracking_enabled = bool(enabled)
            self.tracking_model_path = model_path
            self.tracking_conf = max(0.05, min(0.95, float(conf)))
            self._coco_filter_active = self._is_coco_model(model_path)
            self.tracking_tracker = tracker
            self.preprocess_mode = preprocess_mode

            if not self.tracking_enabled:
                self.tracking_last_error = None
                self.tracking_track_centers.clear()
                return self._tracking_state()

            if self.video_source_mode == "video":
                self._ensure_tracking_model()

            return self._tracking_state()

    def set_video_source(self, path: str | None, loop: bool = True) -> dict[str, Any]:
        with self.lock:
            path_value = (path or "").strip()
            if not path_value:
                self._close_video_capture()
                self.video_source_mode = "synthetic"
                self.video_source_path = None
                self.video_loop = True
                self.video_frame_index = 0
                self.video_last_error = None
                self.video_last_frame = None
                self.tracking_track_centers.clear()
                return self._video_source_state()

            sequence_files = self._discover_sequence_files(path_value)
            if sequence_files is not None:
                if not sequence_files:
                    raise ValueError(f"No readable image frames found in sequence directory: {path_value}")

                first_frame = cv2.imread(sequence_files[0], cv2.IMREAD_COLOR)
                if first_frame is None:
                    raise ValueError(f"Unable to decode first sequence frame: {sequence_files[0]}")

                prepared = self._prepare_video_frame(first_frame)
                self._close_video_capture()
                self.video_sequence_files = sequence_files
                self.video_sequence_index = 1
                self.video_source_mode = "video"
                self.video_source_path = path_value
                self.video_loop = bool(loop)
                self.video_last_frame = prepared
                self.video_frame_index = 1
                self.video_last_error = None
                self.tracking_track_centers.clear()

                if self.tracking_enabled:
                    self._ensure_tracking_model()

                return self._video_source_state()

            candidate = cv2.VideoCapture(path_value)
            if not candidate.isOpened():
                candidate.release()
                raise ValueError(f"Unable to open video source: {path_value}")

            ok, first_frame = candidate.read()
            if not ok or first_frame is None:
                candidate.release()
                raise ValueError(f"Unable to decode frames from video source: {path_value}")

            prepared = self._prepare_video_frame(first_frame)
            self._close_video_capture()
            self.video_capture = candidate
            self.video_source_mode = "video"
            self.video_source_path = path_value
            self.video_loop = bool(loop)
            self.video_last_frame = prepared
            self.video_frame_index = 1
            self.video_last_error = None
            self.tracking_track_centers.clear()

            if self.tracking_enabled:
                self._ensure_tracking_model()

            return self._video_source_state()

    def _discover_sequence_files(self, path_value: str) -> list[str] | None:
        candidate = Path(path_value)
        if not candidate.is_dir():
            return None

        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [
            str(item)
            for item in sorted(candidate.iterdir())
            if item.is_file() and item.suffix.lower() in supported_extensions
        ]
        return files

    def _tracking_active(self) -> bool:
        return self.tracking_enabled and self.video_source_mode == "video"

    # Map of known model stems → HuggingFace (repo_id, remote_filename)
    _HF_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
        "yolov8s-visdrone": ("mshamrai/yolov8s-visdrone", "best.pt"),
        "yolov8l-visdrone": ("mshamrai/yolov8l-visdrone", "best.pt"),
    }

    @staticmethod
    def _try_download_model(path: Path) -> Path | None:
        """Try to download a missing model file from HuggingFace.

        Returns the local Path if download succeeds, None otherwise.
        The file is saved next to where it was requested (usually the cwd).
        """
        stem = path.stem.lower()
        entry = FeatureFirstEngine._HF_MODEL_REGISTRY.get(stem)
        if entry is None:
            return None  # not a known HuggingFace model — let ultralytics handle it

        repo_id, remote_name = entry
        url = f"https://huggingface.co/{repo_id}/resolve/main/{remote_name}"
        dest = path if path.parent != Path(".") else Path(path.name)

        import urllib.request
        try:
            tmp = dest.with_suffix(".tmp")
            print(f"[ffisr] Downloading {stem}.pt from {url} …", flush=True)
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            print(f"[ffisr] Saved to {dest}", flush=True)
            return dest
        except Exception as exc:
            print(f"[ffisr] Auto-download failed for {stem}: {exc}", flush=True)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    @staticmethod
    def _is_coco_model(model_path: str) -> bool:
        """Return True if model_path looks like a stock COCO-pretrained checkpoint.

        VisDrone-finetuned models (e.g. yolov8s-visdrone.pt) and any other
        domain-specific weights are assumed to have correct class names already.
        """
        name = Path(model_path).stem.lower()
        non_coco_markers = ("visdrone", "aerial", "uav", "drone", "rtdetr", "rt-detr")
        if any(m in name for m in non_coco_markers):
            return False
        coco_prefixes = ("yolo", "rtdetr")
        return any(name.startswith(p) for p in coco_prefixes)

    def _ensure_tracking_model(self) -> bool:
        if self.tracking_model is not None:
            return True
        if self._model_loading:
            return False
        self._model_loading = True
        self.tracking_last_error = f"loading {self.tracking_model_path}…"
        target_path = self.tracking_model_path

        def _load() -> None:
            try:
                ultralytics = importlib.import_module("ultralytics")
                # Disable telemetry and hub update checks — avoids rate-limit
                # errors and network calls on air-gapped / FIPS systems.
                try:
                    ultralytics.utils.SETTINGS.update({
                        "sync": False,
                        "checks": False,
                    })
                except Exception:
                    pass
                resolved = Path(target_path)
                if not resolved.exists() and resolved.suffix != "":
                    downloaded = FeatureFirstEngine._try_download_model(resolved)
                    if downloaded is None:
                        raise FileNotFoundError(
                            f"Model file not found: '{target_path}'. "
                            f"Copy the .pt file to the working directory or provide an absolute path."
                        )
                    resolved = downloaded
                _name = resolved.stem.lower()
                model_path_final = str(resolved)
                if "rtdetr" in _name or "rt-detr" in _name:
                    model = ultralytics.RTDETR(model_path_final)
                else:
                    model = ultralytics.YOLO(model_path_final)
            except ImportError:
                with self.lock:
                    self._model_loading = False
                    self.tracking_last_error = "ultralytics is not installed"
                return
            except Exception as exc:
                with self.lock:
                    self._model_loading = False
                    self.tracking_last_error = f"failed to load '{target_path}': {exc}"
                return
            with self.lock:
                # Only promote if the user hasn't switched to a different model
                # while this one was loading.
                if self.tracking_model_path == target_path:
                    self.tracking_model = model
                    self.tracking_last_error = None
                self._model_loading = False

        threading.Thread(target=_load, daemon=True).start()
        return False

    def _close_video_capture(self) -> None:
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.video_sequence_files = []
        self.video_sequence_index = 0

    def _prepare_video_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return frame

    def _read_video_frame(self) -> np.ndarray | None:
        if self.video_sequence_files:
            if self.video_sequence_index >= len(self.video_sequence_files):
                if self.video_loop:
                    self.video_sequence_index = 0
                else:
                    self.video_last_error = "Image sequence has no more readable frames"
                    if self.video_last_frame is None:
                        return None
                    return self.video_last_frame.copy()

            frame_path = self.video_sequence_files[self.video_sequence_index]
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if frame is None:
                self.video_last_error = f"Unable to decode sequence frame: {frame_path}"
                if self.video_last_frame is None:
                    return None
                return self.video_last_frame.copy()

            prepared = self._prepare_video_frame(frame)
            self.video_last_frame = prepared
            self.video_sequence_index += 1
            self.video_frame_index += 1
            self.video_last_error = None
            return prepared.copy()

        if self.video_capture is None:
            return None

        ok, frame = self.video_capture.read()
        if not ok or frame is None:
            if self.video_loop:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.video_capture.read()

        if not ok or frame is None:
            self.video_last_error = "Video source has no more readable frames"
            if self.video_last_frame is None:
                return None
            return self.video_last_frame.copy()

        prepared = self._prepare_video_frame(frame)
        self.video_last_frame = prepared
        self.video_frame_index += 1
        self.video_last_error = None
        return prepared.copy()

    def _run_loop(self) -> None:
        dt_target = 1.0 / float(self.fps)
        while True:
            with self.lock:
                if not self.running:
                    break
            started = time.monotonic()
            with self.lock:
                self._tick_pre_inference()
                frame_for_inference = self._last_inference_frame
                model_snapshot = self.tracking_model if self._tracking_active() else None
                conf_snapshot = self.tracking_conf
                tracker_snapshot = self.tracking_tracker

            if model_snapshot is not None and frame_for_inference is not None:
                try:
                    inference_results = model_snapshot.track(
                        source=frame_for_inference,
                        conf=conf_snapshot,
                        tracker=tracker_snapshot,
                        persist=True,
                        verbose=False,
                    )
                except Exception as exc:
                    inference_results = None
                    with self.lock:
                        self.tracking_last_error = f"tracking inference failed: {exc}"
            else:
                inference_results = None

            with self.lock:
                self._tick_post_inference(inference_results)

            elapsed = time.monotonic() - started
            sleep_s = max(0.0, dt_target - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _update_rf_state(self, sim_ts: float, now: float) -> None:
        in_silence = self.link.in_rf_silence()

        if in_silence and not self.rf_silence_active:
            self.rf_silence_active = True
            self.rf_silence_started_monotonic = now
            self.rf_buffered_tracks.clear()
            self.rf_buffered_events.clear()
            self.rf_reconnect_queue.clear()
            self.rf_reconnect_pending = False
            return

        if not in_silence and self.rf_silence_active:
            self.rf_silence_active = False
            self.rf_last_silence_s = max(0.0, now - self.rf_silence_started_monotonic)
            self.rf_total_silence_s += self.rf_last_silence_s
            self.rf_reconnect_started_monotonic = now
            self.rf_reconnect_latency_ms = 0.0
            self.rf_reconnect_catchup_bytes = 0
            self._queue_reconnect_snapshot(sim_ts=sim_ts)
            self.rf_reconnect_pending = len(self.rf_reconnect_queue) > 0
            self.force_keyframe = True

    def _queue_reconnect_snapshot(self, sim_ts: float) -> None:
        self.rf_reconnect_queue.clear()

        buffered_events = list(self.rf_buffered_events)[-40:]
        for payload in buffered_events:
            queued = dict(payload)
            queued["detail"] = f"[reconnect replay] {queued.get('detail', '')}".strip()
            self.rf_reconnect_queue.append(("event", queued))

        for _track_id, payload in sorted(self.rf_buffered_tracks.items(), key=lambda item: item[0]):
            queued = dict(payload)
            queued["timestamp"] = round(sim_ts, 2)
            self.rf_reconnect_queue.append(("track", queued))

    def _flush_reconnect_queue(self, sim_ts: float, now: float) -> None:
        max_messages_per_tick = 10
        sent = 0

        while self.rf_reconnect_queue and sent < max_messages_per_tick:
            kind, payload = self.rf_reconnect_queue[0]
            packed = serialize_json(payload)
            if not self.link.send("semantic", len(packed)):
                break

            self.rf_reconnect_queue.popleft()
            sent += 1
            self.rf_reconnect_catchup_bytes += len(packed)
            self._record_delivery(now, "semantic", len(packed))

            if kind == "track":
                track_id = int(payload["track_id"])
                bbox = payload.get("bbox")
                if bbox is not None:
                    self.track_prev_bbox[track_id] = list(bbox)
                self.track_last_sent_ts[track_id] = sim_ts
                self._ingest_track_update(payload)
                gt_id = payload.get("gt_id")
                if gt_id is not None and gt_id not in self.detect_ts_by_gt:
                    self.detect_ts_by_gt[gt_id] = sim_ts
            else:
                self.events.appendleft(payload)

        if not self.rf_reconnect_queue and self.rf_reconnect_pending:
            self.rf_reconnect_pending = False
            self.rf_reconnect_latency_ms = max(
                0.0, (time.monotonic() - self.rf_reconnect_started_monotonic) * 1000.0
            )
            self.rf_buffered_tracks.clear()
            self.rf_buffered_events.clear()

    def _tick_pre_inference(self) -> None:
        """First half of the tick: advance state, render frame, send baseline.
        Stores frame + timestamps for the post-inference half.
        """
        self.tick_count += 1
        now = time.monotonic()
        sim_ts = now - self.start_monotonic

        if self._tracking_active():
            self.objects.clear()
        else:
            self._spawn_if_needed(sim_ts)
            self._update_objects(dt=1.0 / self.fps, sim_ts=sim_ts)
        self._update_rf_state(sim_ts=sim_ts, now=now)

        frame = self._render_frame(sim_ts)

        baseline_bytes = self._encode_jpeg(frame, quality=72)
        if self.link.send("baseline", len(baseline_bytes)):
            self.latest_baseline_frame = baseline_bytes
            self.latest_baseline_frame_id += 1
            self._record_delivery(now, "baseline", len(baseline_bytes))

        self._last_tick_frame = frame
        self._last_tick_sim_ts = sim_ts
        self._last_tick_now = now

        if self._tracking_active() and self.tracking_model is not None:
            self._last_inference_frame = self._preprocess_frame(frame)
        else:
            self._last_inference_frame = None

    def _tick_post_inference(self, inference_results: Any) -> None:
        """Second half of the tick: consume inference results, send semantic stream."""
        frame = self._last_tick_frame
        sim_ts = self._last_tick_sim_ts
        now = self._last_tick_now

        if frame is None:
            return

        if self.rf_reconnect_pending:
            self._flush_reconnect_queue(sim_ts=sim_ts, now=now)
            if self.rf_reconnect_pending:
                self._prune_old_samples(now)
                return

        if self.force_keyframe:
            delivered = self._send_keyframe(frame, sim_ts, now)
            if delivered:
                self.last_periodic_keyframe_ts = sim_ts
            # Preserve semantic bandwidth to deliver requested keyframes quickly.
            if self.force_keyframe and not self.link.in_rf_silence():
                self._prune_old_samples(now)
                return

        anomaly_keyframe = self._process_semantic_tracks(frame, sim_ts, now, inference_results=inference_results)

        if anomaly_keyframe or self.force_keyframe:
            delivered = self._send_keyframe(frame, sim_ts, now)
            if delivered:
                self.last_periodic_keyframe_ts = sim_ts

        periodic_interval = self._adaptive_periodic_keyframe_interval()
        if (
            periodic_interval is not None
            and not self.link.in_rf_silence()
            and not self.rf_reconnect_pending
            and not self.force_keyframe
            and (sim_ts - self.last_periodic_keyframe_ts) >= periodic_interval
        ):
            delivered = self._send_keyframe(frame, sim_ts, now)
            if delivered:
                self.last_periodic_keyframe_ts = sim_ts

        self._prune_old_samples(now)

    def _spawn_if_needed(self, sim_ts: float) -> None:
        target = 6
        classes = ["person", "vehicle", "bike"]
        while len(self.objects) < target:
            lane = random.choice([0, 1])
            from_left = random.random() < 0.5
            x = -120.0 if from_left else float(self.width + 40)
            y = random.uniform(50, self.height - 120)
            w = random.uniform(34, 90)
            h = random.uniform(34, 110)
            speed_px = random.uniform(45, 135)
            vx = speed_px if from_left else -speed_px
            vy = random.uniform(-18, 18) if lane == 1 else random.uniform(-6, 6)
            obj = SimObject(
                gt_id=self.next_gt_id,
                cls=random.choice(classes),
                x=x,
                y=y,
                w=w,
                h=h,
                vx=vx,
                vy=vy,
                born_ts=sim_ts,
            )
            self.enter_ts_by_gt[obj.gt_id] = sim_ts
            self._emit_event("enter", track_id=self.gt_to_track.get(obj.gt_id), timestamp=sim_ts, detail=f"{obj.cls} entered AOI")
            self.next_gt_id += 1
            self.objects.append(obj)

    def _update_objects(self, dt: float, sim_ts: float) -> None:
        kept: list[SimObject] = []
        for obj in self.objects:
            if random.random() < 0.04:
                obj.vx *= random.uniform(0.55, 1.35)
                obj.vy *= random.uniform(0.55, 1.35)

            obj.x += obj.vx * dt
            obj.y += obj.vy * dt

            obj.y = max(10.0, min(self.height - obj.h - 10.0, obj.y))

            speed_now = speed(obj.vx, obj.vy)
            if speed_now < 24:
                if obj.loiter_start_ts is None:
                    obj.loiter_start_ts = sim_ts
                elif sim_ts - obj.loiter_start_ts > 5.0:
                    self._emit_event(
                        "loiter",
                        track_id=self.gt_to_track.get(obj.gt_id),
                        timestamp=sim_ts,
                        detail=f"GT {obj.gt_id} loitering",
                    )
                    obj.loiter_start_ts = sim_ts + 999
            else:
                obj.loiter_start_ts = None

            offscreen = obj.x > self.width + 160 or obj.x + obj.w < -160
            if offscreen:
                self.exits_total += 1
                if obj.gt_id not in self.detect_ts_by_gt:
                    self.missed_total += 1
                self._emit_event(
                    "exit",
                    track_id=self.gt_to_track.get(obj.gt_id),
                    timestamp=sim_ts,
                    detail=f"{obj.cls} exited AOI",
                )
                if obj.gt_id in self.gt_to_track:
                    del self.gt_to_track[obj.gt_id]
                continue

            kept.append(obj)

        self.objects = kept

    def _render_frame(self, sim_ts: float) -> np.ndarray:
        video_canvas = self._read_video_frame() if self.video_source_mode == "video" else None
        if video_canvas is None:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            canvas[:] = (20, 23, 27)

            for row in range(0, self.height, 60):
                color = (26, 30, 35) if (row // 60) % 2 == 0 else (23, 27, 31)
                cv2.rectangle(canvas, (0, row), (self.width, min(self.height, row + 60)), color, -1)
        else:
            canvas = video_canvas

        if self._tracking_active():
            return canvas

        for obj in self.objects:
            p1 = (int(obj.x), int(obj.y))
            p2 = (int(obj.x + obj.w), int(obj.y + obj.h))
            cv2.rectangle(canvas, p1, p2, (75, 120, 230), 2)
            cv2.putText(
                canvas,
                f"GT {obj.gt_id} {obj.cls}",
                (p1[0], max(20, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (210, 220, 235),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def _draw_track_overlay(self, canvas: np.ndarray, sim_ts: float) -> None:
        # Keep baseline panel readable by only showing recently updated tracks.
        stale_after_s = 4.0
        for item in self.track_view.values():
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            timestamp = float(item.get("timestamp") or 0.0)
            if sim_ts - timestamp > stale_after_s:
                continue

            x, y, w, h = [int(v) for v in bbox]
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            w = max(1, min(self.width - x, w))
            h = max(1, min(self.height - y, h))

            cv2.rectangle(canvas, (x, y), (x + w, y + h), (65, 186, 255), 2)
            label = f"ID {item.get('track_id', '?')} {item.get('class', 'obj')}"
            cv2.putText(
                canvas,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (232, 241, 250),
                1,
                cv2.LINE_AA,
            )

    def _process_semantic_tracks(self, frame: np.ndarray, sim_ts: float, now: float, inference_results: Any = None) -> bool:
        if self._tracking_active():
            return self._process_model_tracks(frame, sim_ts, now, inference_results=inference_results)

        anomaly_keyframe = False
        visible_tracks: set[int] = set()
        min_update_interval = self._semantic_update_interval()
        rf_silence = self.link.in_rf_silence()

        for obj in self.objects:
            if random.random() < 0.08:
                continue

            should_switch = obj.gt_id in self.gt_to_track and random.random() < 0.015
            if obj.gt_id not in self.gt_to_track or should_switch:
                if should_switch:
                    self.id_switches += 1
                self.gt_to_track[obj.gt_id] = self.next_track_id
                self.next_track_id += 1
                self.track_assignments += 1

            track_id = self.gt_to_track[obj.gt_id]
            visible_tracks.add(track_id)

            last_sent = self.track_last_sent_ts.get(track_id, -999.0)
            if sim_ts - last_sent < min_update_interval:
                continue

            conf = quantize_confidence(random.uniform(0.72, 0.99))
            if random.random() < 0.015:
                conf = quantize_confidence(random.uniform(0.5, 0.61))
            if conf < 0.62:
                self._emit_event(
                    "anomaly",
                    track_id=track_id,
                    timestamp=sim_ts,
                    detail=f"low confidence={conf:.2f}",
                )
                if sim_ts - self.last_auto_keyframe_ts > 4.0:
                    anomaly_keyframe = True
                    self.last_auto_keyframe_ts = sim_ts

            bbox = [int(obj.x), int(obj.y), int(obj.w), int(obj.h)]
            delta, _ = bbox_delta(bbox, self.track_prev_bbox.get(track_id))

            vx = obj.vx + random.uniform(-9, 9)
            vy = obj.vy + random.uniform(-9, 9)
            velocity = quantize_velocity(speed(vx, vy))

            message = {
                "type": "TRACK_UPDATE",
                "track_id": track_id,
                "class": obj.cls,
                "confidence": conf,
                "velocity": velocity,
                "timestamp": round(sim_ts, 2),
                "bbox": bbox if delta is None else None,
                "bbox_delta": delta,
                "gt_id": obj.gt_id,
            }

            if self.tick_count % 12 == 0:
                message["embedding"] = reduced_embedding(track_id, sim_ts, dims=8)

            if rf_silence:
                buffered = dict(message)
                buffered["bbox"] = bbox
                buffered["bbox_delta"] = None
                buffered.pop("embedding", None)
                self.rf_buffered_tracks[track_id] = buffered
                continue

            packed = serialize_json(message)
            if self.link.send("semantic", len(packed)):
                self._record_delivery(now, "semantic", len(packed))
                self.track_prev_bbox[track_id] = bbox
                self.track_last_sent_ts[track_id] = sim_ts
                if obj.gt_id not in self.detect_ts_by_gt:
                    self.detect_ts_by_gt[obj.gt_id] = sim_ts

        self._expire_stale_tracks(visible_tracks=visible_tracks, sim_ts=sim_ts)
        return anomaly_keyframe

    def _process_model_tracks(self, frame: np.ndarray, sim_ts: float, now: float, inference_results: Any = None) -> bool:
        if not self._ensure_tracking_model() or self.tracking_model is None:
            return False

        min_update_interval = self._semantic_update_interval()
        rf_silence = self.link.in_rf_silence()
        visible_tracks: set[int] = set()
        anomaly_keyframe = False

        results = inference_results

        names: dict[int, str] = {}
        boxes = None
        if results:
            first = results[0]
            names_raw = getattr(first, "names", {})
            if isinstance(names_raw, dict):
                names = {int(k): str(v) for k, v in names_raw.items()}
            boxes = getattr(first, "boxes", None)

        if boxes is None:
            self._expire_stale_tracks(visible_tracks=visible_tracks, sim_ts=sim_ts)
            return False

        use_coco_filter = getattr(self, "_coco_filter_active", self._is_coco_model(self.tracking_model_path))

        for box in boxes:
            box_id = getattr(box, "id", None)
            if box_id is None:
                continue

            cls_idx = int(float(box.cls.item())) if getattr(box, "cls", None) is not None else -1
            conf_raw = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0

            if use_coco_filter:
                if cls_idx not in _COCO_AERIAL_FILTER:
                    continue
                cls_name = _COCO_AERIAL_FILTER[cls_idx]
            else:
                cls_name = names.get(cls_idx, f"class_{cls_idx}")

            track_id = int(float(box_id.item()))
            visible_tracks.add(track_id)

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1 = max(0, min(self.width - 1, x1))
            y1 = max(0, min(self.height - 1, y1))
            x2 = max(x1 + 1, min(self.width, x2))
            y2 = max(y1 + 1, min(self.height, y2))
            bbox = [x1, y1, x2 - x1, y2 - y1]

            conf = quantize_confidence(max(0.0, min(1.0, conf_raw)))

            cx = x1 + (bbox[2] / 2.0)
            cy = y1 + (bbox[3] / 2.0)
            prev_center = self.tracking_track_centers.get(track_id)
            velocity_raw = 0.0
            if prev_center is not None:
                dt = max(1e-3, sim_ts - prev_center[2])
                velocity_raw = speed(cx - prev_center[0], cy - prev_center[1]) / dt
            self.tracking_track_centers[track_id] = (cx, cy, sim_ts)
            velocity = quantize_velocity(velocity_raw)

            if conf < 0.5 and sim_ts - self.last_auto_keyframe_ts > 4.0:
                self._emit_event(
                    "anomaly",
                    track_id=track_id,
                    timestamp=sim_ts,
                    detail=f"low confidence={conf:.2f}",
                )
                anomaly_keyframe = True
                self.last_auto_keyframe_ts = sim_ts

            view_message = {
                "type": "TRACK_UPDATE",
                "track_id": track_id,
                "class": cls_name,
                "confidence": conf,
                "velocity": velocity,
                "timestamp": round(sim_ts, 2),
                "bbox": bbox,
                "bbox_delta": None,
                "gt_id": None,
            }
            self._ingest_track_update(view_message)

            last_sent = self.track_last_sent_ts.get(track_id, -999.0)
            if sim_ts - last_sent < min_update_interval:
                continue

            delta, _ = bbox_delta(bbox, self.track_prev_bbox.get(track_id))
            message = dict(view_message)
            message["bbox"] = bbox if delta is None else None
            message["bbox_delta"] = delta

            if self.tick_count % 12 == 0:
                message["embedding"] = reduced_embedding(track_id, sim_ts, dims=8)

            if rf_silence:
                buffered = dict(message)
                buffered["bbox"] = bbox
                buffered["bbox_delta"] = None
                buffered.pop("embedding", None)
                self.rf_buffered_tracks[track_id] = buffered
                continue

            packed = serialize_json(message)
            if self.link.send("semantic", len(packed)):
                self._record_delivery(now, "semantic", len(packed))
                self.track_prev_bbox[track_id] = bbox
                self.track_last_sent_ts[track_id] = sim_ts
                self._ingest_track_update(message)

        for existing_track_id in list(self.tracking_track_centers.keys()):
            if existing_track_id not in visible_tracks:
                self.tracking_track_centers.pop(existing_track_id, None)

        self.tracking_last_error = None
        self._expire_stale_tracks(visible_tracks=visible_tracks, sim_ts=sim_ts)
        return anomaly_keyframe

    def _expire_stale_tracks(self, visible_tracks: set[int], sim_ts: float) -> None:
        to_remove: list[int] = []
        stale_ttl = max(1.8, self._semantic_update_interval() * 10.0)
        for track_id, info in self.track_view.items():
            if track_id in visible_tracks:
                info["visible"] = True
                continue
            info["visible"] = False
            if sim_ts - info["timestamp"] > stale_ttl:
                to_remove.append(track_id)

        for track_id in to_remove:
            self.track_view.pop(track_id, None)
            self.track_prev_bbox.pop(track_id, None)
            self.track_last_sent_ts.pop(track_id, None)

    def _ingest_track_update(self, message: dict[str, Any]) -> None:
        track_id = int(message["track_id"])
        existing = self.track_view.get(track_id)

        if existing and message.get("bbox_delta") is not None and existing.get("bbox"):
            prev = existing["bbox"]
            delta = message["bbox_delta"]
            bbox = [prev[i] + delta[i] for i in range(4)]
        else:
            if existing:
                bbox = message.get("bbox") or existing.get("bbox") or [0, 0, 0, 0]
            else:
                bbox = message.get("bbox") or [0, 0, 0, 0]

        self.track_view[track_id] = {
            "track_id": track_id,
            "class": message.get("class", "unknown"),
            "confidence": message.get("confidence", 0.0),
            "velocity": message.get("velocity", 0.0),
            "timestamp": message.get("timestamp", 0.0),
            "bbox": bbox,
            "gt_id": message.get("gt_id"),
            "embedding_dims": len(message.get("embedding", [])),
        }

    def _send_keyframe(self, frame: np.ndarray, sim_ts: float, now: float) -> bool:
        annotated = frame.copy()

        if self.force_keyframe:
            target_size = (max(640, (self.width * 3) // 4), max(360, (self.height * 3) // 4))
            quality = 50
        else:
            if self.link.link_kbps >= 3000:
                target_size = (max(720, (self.width * 4) // 5), max(400, (self.height * 4) // 5))
                quality = 48
            elif self.link.link_kbps >= 1500:
                target_size = (max(640, (self.width * 3) // 4), max(360, (self.height * 3) // 4))
                quality = 42
            else:
                target_size = (max(420, self.width // 2), max(240, self.height // 2))
                quality = 35

        encoded_frame = cv2.resize(
            annotated,
            target_size,
            interpolation=cv2.INTER_AREA,
        )
        payload = self._encode_jpeg(encoded_frame, quality=quality)
        envelope_size = len(payload) + 36
        delivered = False
        if self.link.send("semantic", envelope_size):
            self.latest_keyframe = payload
            self.latest_keyframe_id += 1
            self._record_delivery(now, "semantic", envelope_size)
            delivered = True
            if self.force_keyframe or self.pending_keyframe_track is not None:
                self._emit_event(
                    "anomaly" if self.pending_keyframe_track is None else "enter",
                    track_id=self.pending_keyframe_track,
                    timestamp=sim_ts,
                    detail="keyframe delivered",
                )

        if delivered or not self.force_keyframe:
            self.pending_keyframe_track = None
            self.force_keyframe = False

        return delivered

    def _semantic_update_interval(self) -> float:
        link_kbps = self.link.link_kbps
        if link_kbps <= 300:
            base = 0.45
        elif link_kbps <= 600:
            base = 0.28
        elif link_kbps <= 1000:
            base = 0.18
        elif link_kbps <= 2000:
            base = 0.12
        elif link_kbps <= 5000:
            base = 0.08
        else:
            base = 0.06

        stats = self.link.stats().get("semantic", {})
        attempted = int(stats.get("attempted_packets", 0))
        dropped = int(stats.get("dropped_packets", 0))
        drop_ratio = (dropped / float(attempted)) if attempted > 30 else 0.0

        if drop_ratio > 0.35:
            base *= 1.8
        elif drop_ratio > 0.2:
            base *= 1.35
        elif drop_ratio < 0.05 and link_kbps >= 2000:
            base *= 0.75

        return max(0.04, min(0.7, base))

    def _adaptive_periodic_keyframe_interval(self) -> float | None:
        tracking_video = self._tracking_active()
        if self.link.link_kbps < 900 and not tracking_video:
            return None

        if self.link.link_kbps >= 5000:
            interval = 0.55 if tracking_video else 0.8
        elif self.link.link_kbps >= 2500:
            interval = 0.9 if tracking_video else 1.2
        elif self.link.link_kbps >= 1500:
            interval = 1.4 if tracking_video else 1.8
        elif self.link.link_kbps >= 900:
            interval = 2.0 if tracking_video else 2.8
        else:
            interval = 3.4

        stats = self.link.stats().get("semantic", {})
        attempted = int(stats.get("attempted_packets", 0))
        dropped = int(stats.get("dropped_packets", 0))
        drop_ratio = (dropped / float(attempted)) if attempted > 30 else 0.0

        if drop_ratio > 0.2:
            interval *= 1.8
        elif drop_ratio > 0.1:
            interval *= 1.35

        return max(0.6, min(5.0, interval))

    def _emit_event(self, event_type: str, track_id: int | None, timestamp: float, detail: str) -> None:
        payload = {
            "type": "EVENT",
            "event_type": event_type,
            "track_id": track_id,
            "timestamp": round(timestamp, 2),
            "detail": detail,
        }

        if self.link.in_rf_silence():
            self.rf_buffered_events.append(payload)
            return

        packed = serialize_json(payload)
        if self.link.send("semantic", len(packed)):
            self.events.appendleft(payload)
            self._record_delivery(time.monotonic(), "semantic", len(packed))

    def _encode_jpeg(self, image: np.ndarray, quality: int) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            return b""
        return encoded.tobytes()

    def _record_delivery(self, now: float, channel: str, nbytes: int) -> None:
        self.delivered_samples.append((now, channel, nbytes))

    def _prune_old_samples(self, now: float) -> None:
        cutoff = now - 180.0
        if not self.delivered_samples:
            return
        idx = 0
        for idx, sample in enumerate(self.delivered_samples):
            if sample[0] >= cutoff:
                break
        else:
            self.delivered_samples.clear()
            return
        if idx > 0:
            self.delivered_samples = self.delivered_samples[idx:]

    def _percentile(self, values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        pos = int(round((q / 100.0) * (len(ordered) - 1)))
        return ordered[max(0, min(pos, len(ordered) - 1))]

    def _bandwidth_metrics(self) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = max(1.0, now - self.start_monotonic)
        delivered_total = sum(sample[2] for sample in self.delivered_samples)
        avg_kbps = (delivered_total * 8.0 / elapsed) / 1000.0

        sec_buckets: dict[int, int] = {}
        for ts, _channel, nbytes in self.delivered_samples:
            sec = int(ts)
            sec_buckets[sec] = sec_buckets.get(sec, 0) + nbytes

        per_sec_kbps = [(v * 8.0) / 1000.0 for v in sec_buckets.values()]
        p95_kbps = self._percentile(per_sec_kbps, 95)

        total_mb = delivered_total / (1024.0 * 1024.0)
        total_mb_per_min = total_mb / max(elapsed / 60.0, 1e-6)

        semantic_bytes = sum(sample[2] for sample in self.delivered_samples if sample[1] == "semantic")
        baseline_bytes = sum(sample[2] for sample in self.delivered_samples if sample[1] == "baseline")
        semantic_kbps = (semantic_bytes * 8.0 / elapsed) / 1000.0
        baseline_kbps = (baseline_bytes * 8.0 / elapsed) / 1000.0

        savings_pct = 0.0
        if baseline_bytes > 0:
            savings_pct = max(0.0, min(100.0, (1.0 - (semantic_bytes / float(baseline_bytes))) * 100.0))

        return {
            "avg_kbps": round(avg_kbps, 2),
            "p95_kbps": round(p95_kbps, 2),
            "total_mb_per_min": round(total_mb_per_min, 3),
            "semantic_mb": round(semantic_bytes / (1024.0 * 1024.0), 3),
            "baseline_mb": round(baseline_bytes / (1024.0 * 1024.0), 3),
            "semantic_kbps": round(semantic_kbps, 2),
            "baseline_kbps": round(baseline_kbps, 2),
            "bandwidth_savings_pct": round(savings_pct, 1),
            "energy_proxy": round((delivered_total * 8.0) / 1_000_000.0, 3),
            "power_proxy_semantic_mbits": round((semantic_bytes * 8.0) / 1_000_000.0, 3),
            "power_proxy_baseline_mbits": round((baseline_bytes * 8.0) / 1_000_000.0, 3),
            "power_savings_pct": round(savings_pct, 1),
        }

    def _utility_metrics(self) -> dict[str, Any]:
        latencies = []
        for gt_id, enter_ts in self.enter_ts_by_gt.items():
            detect_ts = self.detect_ts_by_gt.get(gt_id)
            if detect_ts is None:
                continue
            latencies.append(max(0.0, detect_ts - enter_ts))

        avg_ttd = sum(latencies) / len(latencies) if latencies else 0.0
        p95_ttd = self._percentile(latencies, 95)

        continuity = 1.0
        if self.track_assignments > 0:
            continuity = max(0.0, 1.0 - (self.id_switches / float(self.track_assignments)))

        miss_rate = self.missed_total / float(self.exits_total) if self.exits_total else 0.0

        return {
            "time_to_detect_avg_s": round(avg_ttd, 2),
            "time_to_detect_p95_s": round(p95_ttd, 2),
            "track_continuity": round(continuity, 3),
            "id_switches": self.id_switches,
            "miss_rate": round(miss_rate, 3),
            "rf_total_silence_s": round(self.rf_total_silence_s, 2),
            "rf_last_silence_s": round(self.rf_last_silence_s, 2),
            "rf_resync_latency_ms": round(self.rf_reconnect_latency_ms, 1),
            "rf_catchup_kb": round(self.rf_reconnect_catchup_bytes / 1024.0, 2),
        }

    def _rf_link_state(self) -> dict[str, Any]:
        return {
            "active": self.link.in_rf_silence(),
            "remaining_s": round(self.link.rf_silence_remaining(), 1),
            "reconnect_pending": self.rf_reconnect_pending,
            "buffered_tracks": len(self.rf_buffered_tracks),
            "buffered_events": len(self.rf_buffered_events),
            "catchup_queue": len(self.rf_reconnect_queue),
        }

    def _video_source_state(self) -> dict[str, Any]:
        return {
            "mode": self.video_source_mode,
            "path": self.video_source_path,
            "loop": self.video_loop,
            "frame_index": self.video_frame_index,
            "error": self.video_last_error,
            "active": self.video_source_mode == "video" and (
                self.video_capture is not None or bool(self.video_sequence_files)
            ),
        }

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply detection-oriented preprocessing before inference.

        visible: CLAHE on luminance channel to lift local contrast in hazy/flat
                 aerial imagery, mild bilateral denoise to reduce JPEG noise
                 without blurring edges, and gentle gamma lift for underexposed
                 sequences.

        swir:    SWIR sensors produce single-channel 16-bit output with strong
                 fixed-pattern noise and a heavily skewed histogram. Pipeline:
                 convert to float32, percentile stretch to [0,255], apply
                 aggressive CLAHE (clip=4), bilateral denoise, then replicate
                 to 3-channel BGR so YOLO receives the expected input shape.
                 No gamma correction — SWIR radiometry is linear.
        """
        mode = self.preprocess_mode
        if mode == "none":
            return frame

        if mode == "swir":
            # Accept single-channel uint8/uint16 or 3-channel already-converted.
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            # Percentile stretch to use full uint8 range robustly.
            p2 = float(np.percentile(gray, 2))
            p98 = float(np.percentile(gray, 98))
            if p98 > p2:
                gray = np.clip((gray.astype(np.float32) - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
            # Aggressive CLAHE for fixed-pattern noise suppression.
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            # Bilateral filter: preserves edges while removing sensor noise.
            gray = cv2.bilateralFilter(gray, d=7, sigmaColor=30, sigmaSpace=30)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Default: visible mode — four-stage pipeline tuned for aerial ISR.

        # Stage 1: Dark channel dehazing (fast approximation).
        # Atmospheric scattering at altitude compresses low-end contrast.
        # Dark channel prior: min over a small patch across channels gives
        # a rough atmospheric veil estimate; subtracting it restores depth contrast.
        bgr_f = frame.astype(np.float32) / 255.0
        dark = np.min(bgr_f, axis=2)
        kernel_dh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark = cv2.erode(dark, kernel_dh)
        atmos = float(np.percentile(dark, 99))
        atmos = max(0.1, min(0.95, atmos))
        dehazed = np.clip((bgr_f - atmos) / max(1e-3, 1.0 - atmos), 0.0, 1.0)
        frame = (dehazed * 255.0).astype(np.uint8)

        # Stage 2: Adaptive gamma — target mean luminance ~128 instead of fixed γ.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        mean_l = float(np.mean(l_ch))
        if mean_l > 1.0:
            gamma = np.log(128.0 / 255.0) / np.log(mean_l / 255.0)
            gamma = float(np.clip(gamma, 0.5, 2.0))
        else:
            gamma = 1.0
        table = np.array([min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)], dtype=np.uint8)
        l_ch = cv2.LUT(l_ch, table)

        # Stage 3: CLAHE on L — finer 16×16 tiles for small-object aerial scenes.
        # Smaller tiles give tighter local adaptation without halo artefacts.
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Stage 4: Bilateral denoise then unsharp mask.
        # Denoise first (d=5 is fast; removes JPEG block noise).
        result = cv2.bilateralFilter(result, d=5, sigmaColor=20, sigmaSpace=20)
        # Unsharp mask: sharpen edges blurred by denoise and atmospheric scatter.
        # Recovers fine structure (pedestrians, bikes) near the detector resolution limit.
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.5)
        result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
        return result

    def _tracking_state(self) -> dict[str, Any]:
        return {
            "enabled": self.tracking_enabled,
            "active": self._tracking_active(),
            "model_path": self.tracking_model_path,
            "conf": self.tracking_conf,
            "tracker": self.tracking_tracker,
            "preprocess_mode": self.preprocess_mode,
            "model_loading": self._model_loading,
            "error": self.tracking_last_error,
        }

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            now_ts = time.monotonic() - self.start_monotonic
            stale_cutoff = max(1.8, self._semantic_update_interval() * 3.0)
            tracks = sorted(
                (
                    item for item in self.track_view.values()
                    if item.get("visible") is True
                    or (
                        item.get("visible") is None
                        and (now_ts - float(item.get("timestamp", 0))) < stale_cutoff
                    )
                ),
                key=lambda item: item["track_id"],
            )
            events = list(self.events)[:20]
            bandwidth = self._bandwidth_metrics()
            utility = self._utility_metrics()
            transport = self.link.stats()
            rf_link = self._rf_link_state()
            video_source = self._video_source_state()
            tracking = self._tracking_state()
            return {
                "tracks": tracks,
                "events": events,
                "bandwidth": bandwidth,
                "utility": utility,
                "transport": transport,
                "rf_link": rf_link,
                "video_source": video_source,
                "tracking": tracking,
                "baseline_frame_id": self.latest_baseline_frame_id,
                "keyframe_frame_id": self.latest_keyframe_id,
            }

    def get_baseline_frame(self) -> bytes:
        with self.lock:
            return self.latest_baseline_frame

    def get_keyframe(self) -> bytes:
        with self.lock:
            return self.latest_keyframe

    def get_summary(self) -> dict[str, Any]:
        with self.lock:
            bandwidth = self._bandwidth_metrics()
            utility = self._utility_metrics()
            transport = self.link.stats()

            summary_lines = [
                "Feature-First ISR Demo Summary",
                f"Avg throughput: {bandwidth['avg_kbps']} kbps (p95 {bandwidth['p95_kbps']} kbps)",
                f"Channel split: semantic {bandwidth['semantic_kbps']} kbps vs baseline {bandwidth['baseline_kbps']} kbps",
                f"Estimated link + power savings from semantic stream: {bandwidth['bandwidth_savings_pct']}%",
                f"Data rate: {bandwidth['total_mb_per_min']} MB/min",
                f"Semantic payload: {bandwidth['semantic_mb']} MB vs Baseline payload: {bandwidth['baseline_mb']} MB",
                f"Time-to-detect avg/p95: {utility['time_to_detect_avg_s']}s / {utility['time_to_detect_p95_s']}s",
                f"Track continuity: {utility['track_continuity']} (ID switches={utility['id_switches']})",
                f"Miss rate: {utility['miss_rate']}",
                f"RF silence total/last: {utility['rf_total_silence_s']}s / {utility['rf_last_silence_s']}s",
                f"RF reacquisition latency: {utility['rf_resync_latency_ms']} ms (catch-up {utility['rf_catchup_kb']} KB)",
                f"Energy proxy (Mbits sent): {bandwidth['energy_proxy']}",
            ]

            return {
                "summary_lines": summary_lines,
                "bandwidth": bandwidth,
                "utility": utility,
                "transport": transport,
            }
