const baselineCanvas = document.getElementById("baselineCanvas");
const baselineCtx = baselineCanvas.getContext("2d");
let displayedBaseline = new Image();
let pendingBaseline = null;

function _promoteBaseline(img) {
  displayedBaseline = img;
  pendingBaseline = null;
  baselineCtx.drawImage(displayedBaseline, 0, 0, baselineCanvas.width, baselineCanvas.height);
}
const baselineStatus = document.getElementById("baselineStatus");
const semanticStatus = document.getElementById("semanticStatus");
const trackBody = document.getElementById("trackBody");
const eventList = document.getElementById("eventList");
const metricsPanel = document.getElementById("metricsPanel");
const summaryOutput = document.getElementById("summaryOutput");
const canvas = document.getElementById("semanticCanvas");
const ctx = canvas.getContext("2d");

const linkKbps = document.getElementById("linkKbps");
const packetLoss = document.getElementById("packetLoss");
const semanticRatio = document.getElementById("semanticRatio");
const rfSilenceSeconds = document.getElementById("rfSilenceSeconds");
const videoSourcePath = document.getElementById("videoSourcePath");
const videoLoop = document.getElementById("videoLoop");
const trackingEnabled = document.getElementById("trackingEnabled");
const trackingModelSelect = document.getElementById("trackingModelSelect");
const trackingModelCustomRow = document.getElementById("trackingModelCustomRow");
const trackingModelPath = document.getElementById("trackingModelPath");
const trackingTracker = document.getElementById("trackingTracker");
const trackingConf = document.getElementById("trackingConf");
const applyLink = document.getElementById("applyLink");
const triggerRFSilence = document.getElementById("triggerRFSilence");
const applyVideoSourceBtn = document.getElementById("applyVideoSource");
const useSyntheticSourceBtn = document.getElementById("useSyntheticSource");
const applyTrackingConfigBtn = document.getElementById("applyTrackingConfig");
const requestKeyframe = document.getElementById("requestKeyframe");
const trackSelect = document.getElementById("trackSelect");
const refreshSummary = document.getElementById("refreshSummary");

let selectedTrackId = null;
let batchPollTimer = null;

// Mark an element as user-touched so polling won't overwrite it.
function markTouched(el) { el._userTouched = true; }
function clearTouched(...els) { for (const el of els) el._userTouched = false; }

// Wire touch listeners for every input that polling syncs from state.
for (const el of [videoSourcePath, videoLoop, trackingEnabled, trackingTracker,
                   trackingModelSelect, trackingModelPath, trackingConf]) {
  if (el) el.addEventListener("input",  () => markTouched(el));
  if (el) el.addEventListener("change", () => markTouched(el));
}

function setPill(el, text, kind) {
  el.textContent = text;
  el.classList.remove("ok", "warn", "error", "rf");
  if (kind) el.classList.add(kind);
}
let tracks = [];
let lastBaselineId = -1;
let lastKeyframeId = -1;
let lastSemanticTrackTs = -1;
let lastSemanticWallClockMs = Date.now();

let displayedKeyframe = new Image();
let pendingKeyframe = null;

function _promoteKeyframe(img) {
  displayedKeyframe = img;
  pendingKeyframe = null;
}

function drawPlaceholder() {
  ctx.fillStyle = "#040810";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#4a6070";
  ctx.font = "13px IBM Plex Mono, Menlo, monospace";
  ctx.fillText("awaiting keyframe", 16, 28);
}

function resolvedModelPath() {
  const sel = trackingModelSelect ? trackingModelSelect.value : "yolo11n.pt";
  if (sel === "custom") {
    return (trackingModelPath ? trackingModelPath.value : "").trim() || "yolo11n.pt";
  }
  return sel;
}

async function applyTrackingConfig() {
  applyTrackingConfigBtn.disabled = true;
  try {
    const preprocessEl = document.getElementById("preprocessMode");
    const confVal = trackingConf ? Math.max(0.05, Math.min(0.95, parseFloat(trackingConf.value) || 0.25)) : 0.25;
    const payload = {
      enabled: Boolean(trackingEnabled.checked),
      model_path: resolvedModelPath(),
      conf: confVal,
      tracker: trackingTracker.value || "bytetrack.yaml",
      preprocess_mode: preprocessEl ? preprocessEl.value : "visible",
    };

    const res = await fetch("/api/tracking_config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      throw new Error(`tracking config failed: ${res.status}`);
    }
    const data = await res.json();
    const tracking = data.tracking || {};
    if (tracking.error) {
      setPill(semanticStatus, `tracking: ${tracking.error}`, "warn");
    } else if (tracking.active) {
      setPill(semanticStatus, `tracking active (${tracking.tracker})`, "ok");
    } else {
      setPill(semanticStatus, `tracking ${tracking.enabled ? "enabled" : "disabled"}`, null);
    }
    clearTouched(trackingEnabled, trackingModelSelect, trackingModelPath, trackingTracker, trackingConf);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setPill(semanticStatus, message, "error");
  } finally {
    applyTrackingConfigBtn.disabled = false;
  }
}

async function setVideoSource(path, loop) {
  const res = await fetch("/api/video_source", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, loop }),
  });

  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    const detail = payload && payload.error ? ` (${payload.error})` : "";
    throw new Error(`video source request failed: ${res.status}${detail}`);
  }

  return res.json();
}

async function applyVideoSource() {
  applyVideoSourceBtn.disabled = true;
  try {
    const path = (videoSourcePath.value || "").trim();
    if (!path) {
      throw new Error("video path is empty");
    }
    await setVideoSource(path, Boolean(videoLoop.checked));
    clearTouched(videoSourcePath, videoLoop);
    setPill(baselineStatus, "video source set", "ok");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setPill(baselineStatus, message, "error");
  } finally {
    applyVideoSourceBtn.disabled = false;
  }
}

async function useSyntheticSource() {
  useSyntheticSourceBtn.disabled = true;
  try {
    await setVideoSource(null, true);
    clearTouched(videoSourcePath, videoLoop);
    setPill(baselineStatus, "synthetic", "ok");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setPill(baselineStatus, message, "error");
  } finally {
    useSyntheticSourceBtn.disabled = false;
  }
}

async function triggerRFSilenceWindow() {
  triggerRFSilence.disabled = true;
  const seconds = Number(rfSilenceSeconds.value);
  try {
    const res = await fetch("/api/rf_silence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ duration_s: seconds }),
    });
    if (!res.ok) {
      throw new Error(`rf silence request failed: ${res.status}`);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setPill(semanticStatus, message, "error");
  } finally {
    triggerRFSilence.disabled = false;
  }
}

function renderSemanticFrame() {
  if (displayedKeyframe.complete && displayedKeyframe.naturalWidth > 0) {
    ctx.drawImage(displayedKeyframe, 0, 0, canvas.width, canvas.height);
  } else {
    drawPlaceholder();
  }

  for (const track of tracks) {
    if (!track.bbox || track.bbox.length !== 4) continue;
    const isSelected = selectedTrackId === track.track_id;
    const [x, y, w, h] = track.bbox;
    const [r, g, b] = isSelected ? [57, 229, 128] : [255, 180, 104];
    ctx.strokeStyle = `rgb(${r},${g},${b})`;
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.strokeRect(x, y, w, h);

    const text = `ID ${track.track_id} ${track.class} ${(track.confidence * 100).toFixed(0)}%`;
    ctx.font = `13px IBM Plex Mono, Menlo, monospace`;
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = `rgba(10,17,24,0.74)`;
    ctx.fillRect(x, Math.max(0, y - 20), tw + 10, 20);
    ctx.fillStyle = `rgb(219,237,248)`;
    ctx.fillText(text, x + 5, Math.max(14, y - 5));
  }
}

function updateTrackTable(nextTracks) {
  trackBody.innerHTML = "";
  for (const item of nextTracks) {
    const tr = document.createElement("tr");
    tr.dataset.trackId = String(item.track_id);
    if (selectedTrackId === item.track_id) {
      tr.classList.add("selected");
    }
    tr.innerHTML = `
      <td>${item.track_id}</td>
      <td>${item.class}</td>
      <td>${Number(item.confidence).toFixed(2)}</td>
      <td>${Number(item.velocity).toFixed(1)}</td>
      <td>${Number(item.timestamp).toFixed(1)}</td>
    `;
    tr.onclick = () => {
      selectedTrackId = item.track_id;
      trackSelect.value = String(item.track_id);
      updateTrackTable(tracks);
      renderSemanticFrame();
    };
    trackBody.appendChild(tr);
  }
}

function updateEvents(events) {
  eventList.innerHTML = "";
  for (const ev of events) {
    const li = document.createElement("li");
    li.textContent = `[${Number(ev.timestamp).toFixed(1)}s] ${ev.event_type.toUpperCase()} track=${ev.track_id ?? "-"} ${ev.detail ?? ""}`;
    eventList.appendChild(li);
  }
}

function updateMetrics(state) {
  const b = state.bandwidth;
  const u = state.utility;
  const t = state.transport;
  const tracking = state.tracking || {};
  const semanticDrop = t.semantic ? `${t.semantic.dropped_packets}/${t.semantic.attempted_packets}` : "0/0";
  const baselineDrop = t.baseline ? `${t.baseline.dropped_packets}/${t.baseline.attempted_packets}` : "0/0";
  const semanticKbps = Number(b.semantic_kbps ?? 0).toFixed(2);
  const baselineKbps = Number(b.baseline_kbps ?? 0).toFixed(2);
  const savingsPct = Number(b.bandwidth_savings_pct ?? 0).toFixed(1);

  const savingsPctEl = document.getElementById("savingsPct");
  const savingsFillEl = document.getElementById("savingsFill");
  const savingsDetailEl = document.getElementById("savingsDetail");
  if (savingsPctEl) savingsPctEl.textContent = `${savingsPct}%`;
  if (savingsFillEl) savingsFillEl.style.width = `${Math.min(100, parseFloat(savingsPct))}%`;
  if (savingsDetailEl) savingsDetailEl.textContent = `${semanticKbps} kbps vs ${baselineKbps} kbps baseline`;
  const semanticPower = Number(b.power_proxy_semantic_mbits ?? 0).toFixed(3);
  const baselinePower = Number(b.power_proxy_baseline_mbits ?? 0).toFixed(3);
  const powerSavings = Number(b.power_savings_pct ?? 0).toFixed(1);
  const rfTotal = Number(u.rf_total_silence_s ?? 0).toFixed(1);
  const rfLast = Number(u.rf_last_silence_s ?? 0).toFixed(1);
  const rfResyncMs = Number(u.rf_resync_latency_ms ?? 0).toFixed(1);
  const rfCatchupKb = Number(u.rf_catchup_kb ?? 0).toFixed(2);
  const trackingMode = tracking.active ? "active" : tracking.enabled ? "enabled (waiting for video)" : "disabled";
  const trackingError = tracking.error ? ` • ${tracking.error}` : "";

  metricsPanel.innerHTML = `
    <div><strong>Bandwidth</strong>: avg ${b.avg_kbps} kbps, p95 ${b.p95_kbps} kbps</div>
    <div><strong>Comparison</strong>: semantic ${semanticKbps} kbps vs baseline ${baselineKbps} kbps</div>
    <div><strong>Savings</strong>: ${savingsPct}% less link usage for semantic stream</div>
    <div><strong>Power proxy</strong>: semantic ${semanticPower} Mbits vs baseline ${baselinePower} Mbits (savings ${powerSavings}%)</div>
    <div><strong>RF silence</strong>: total ${rfTotal}s, last ${rfLast}s, resync ${rfResyncMs} ms, catch-up ${rfCatchupKb} KB</div>
    <div><strong>Tracking</strong>: ${trackingMode} • model ${tracking.model_path || "-"} • tracker ${tracking.tracker || "-"}${trackingError}</div>
    <div><strong>Total</strong>: ${b.total_mb_per_min} MB/min (semantic ${b.semantic_mb} MB, baseline ${b.baseline_mb} MB)</div>
    <div><strong>Utility</strong>: TTD avg ${u.time_to_detect_avg_s}s, p95 ${u.time_to_detect_p95_s}s</div>
    <div><strong>Continuity</strong>: ${u.track_continuity} (ID switches=${u.id_switches})</div>
    <div><strong>Miss rate</strong>: ${u.miss_rate}</div>
    <div><strong>Drops</strong>: semantic ${semanticDrop} | baseline ${baselineDrop}</div>
    <div><strong>Energy proxy</strong>: ${b.energy_proxy} Mbits sent</div>
  `;
}

async function pollState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`state request failed: ${response.status}`);
    }
    const state = await response.json();

    tracks = state.tracks || [];
    const rf = state.rf_link || {};
    const videoSource = state.video_source || {};
    const tracking = state.tracking || {};
    updateTrackTable(tracks);
    updateEvents(state.events || []);
    updateMetrics(state);

    if (state.baseline_frame_id !== lastBaselineId) {
      lastBaselineId = state.baseline_frame_id;
      const incomingBaseline = new Image();
      pendingBaseline = incomingBaseline;
      incomingBaseline.onload = () => { if (pendingBaseline === incomingBaseline) _promoteBaseline(incomingBaseline); };
      incomingBaseline.onerror = () => { if (pendingBaseline === incomingBaseline) pendingBaseline = null; };
      incomingBaseline.src = `/api/baseline_frame?i=${lastBaselineId}`;
      if (videoSource.mode === "video") {
        setPill(baselineStatus, `video #${videoSource.frame_index || lastBaselineId}`, "ok");
      } else {
        setPill(baselineStatus, `frame #${lastBaselineId}`, "ok");
      }
    }

    if (Object.keys(tracking).length > 0) {
      if (!trackingEnabled._userTouched) {
        trackingEnabled.checked = Boolean(tracking.enabled);
      }
      if (!trackingModelSelect._userTouched) {
        if (tracking.model_path && trackingModelSelect) {
          const opt = Array.from(trackingModelSelect.options).find(o => o.value === tracking.model_path);
          if (opt) {
            trackingModelSelect.value = tracking.model_path;
            trackingModelCustomRow.style.display = "none";
          } else {
            trackingModelSelect.value = "custom";
            trackingModelCustomRow.style.display = "";
            if (!trackingModelPath._userTouched && trackingModelPath) {
              trackingModelPath.value = tracking.model_path;
            }
          }
        }
        if (!trackingTracker._userTouched && tracking.tracker) {
          trackingTracker.value = tracking.tracker;
        }
      if (trackingConf && !trackingConf._userTouched && tracking.conf != null) {
        trackingConf.value = Number(tracking.conf).toFixed(2);
      }
      }
    }

    if (videoSource.mode === "video" && videoSource.path) {
      if (!videoSourcePath._userTouched) {
        videoSourcePath.value = videoSource.path;
      }
      if (!videoLoop._userTouched) {
        videoLoop.checked = Boolean(videoSource.loop);
      }
      if (videoSource.error) {
        setPill(baselineStatus, `warn: ${videoSource.error}`, "warn");
      }
    }

    if (rf.active) {
      setPill(baselineStatus, `RF SILENCE ${Number(rf.remaining_s || 0).toFixed(1)}s`, "rf");
    }

    let newestTrackTs = -1;
    for (const track of tracks) {
      newestTrackTs = Math.max(newestTrackTs, Number(track.timestamp || 0));
    }
    if (newestTrackTs > lastSemanticTrackTs) {
      lastSemanticTrackTs = newestTrackTs;
      lastSemanticWallClockMs = Date.now();
    }

    if (state.keyframe_frame_id !== lastKeyframeId) {
      lastKeyframeId = state.keyframe_frame_id;
      const incoming = new Image();
      pendingKeyframe = incoming;
      incoming.onload = () => { if (pendingKeyframe === incoming) _promoteKeyframe(incoming); };
      incoming.onerror = () => { if (pendingKeyframe === incoming) pendingKeyframe = null; };
      incoming.src = `/api/keyframe?i=${lastKeyframeId}`;
      setPill(semanticStatus, `keyframe #${lastKeyframeId} \u2022 tracks ${tracks.length}`, "ok");
    } else {
      const sinceUpdate = Math.max(0, (Date.now() - lastSemanticWallClockMs) / 1000);
      const stale = sinceUpdate > 3.0;
      setPill(semanticStatus, `tracks ${tracks.length} • ${sinceUpdate.toFixed(1)}s ago`, stale ? "warn" : "ok");
    }

    if (rf.active) {
      setPill(semanticStatus, `RF SILENCE ${Number(rf.remaining_s || 0).toFixed(1)}s • buf ${rf.buffered_tracks || 0}`, "rf");
      setPill(baselineStatus, `RF SILENCE ${Number(rf.remaining_s || 0).toFixed(1)}s`, "rf");
    } else if (rf.reconnect_pending) {
      setPill(semanticStatus, `REACQUIRING • queue ${rf.catchup_queue || 0}`, "warn");
    } else if (tracking.model_loading) {
      setPill(semanticStatus, `loading ${tracking.model_path}\u2026`, "warn");
    } else if (tracking.error) {
      setPill(semanticStatus, `${semanticStatus.textContent} • ${tracking.error}`, "warn");
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setPill(baselineStatus, "state fetch failed", "error");
    setPill(semanticStatus, message, "error");
  }
}

async function applyLinkProfile() {
  const payload = {
    link_kbps: Number(linkKbps.value),
    packet_loss: Number(packetLoss.value),
    semantic_ratio: Number(semanticRatio.value),
  };
  await fetch("/api/link_profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function sendKeyframeRequest() {
  const raw = trackSelect.value.trim();
  const payload = {
    track_id: raw === "" ? selectedTrackId : Number(raw),
  };
  await fetch("/api/request_keyframe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function loadSummary() {
  refreshSummary.disabled = true;
  refreshSummary.textContent = "Refreshing…";
  try {
    const res = await fetch("/api/summary", { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`summary request failed: ${res.status}`);
    }
    const data = await res.json();
    const stamp = new Date().toLocaleTimeString();
    summaryOutput.textContent = `[updated ${stamp}]\n${data.summary_lines.join("\n")}`;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    summaryOutput.textContent = `Summary refresh failed: ${message}`;
  } finally {
    refreshSummary.disabled = false;
    refreshSummary.textContent = "Summary slide text";
  }
}

for (const btn of document.querySelectorAll("button[data-preset]")) {
  btn.onclick = async () => {
    const preset = Number(btn.dataset.preset);
    linkKbps.value = String(preset);
    if (preset <= 300) {
      semanticRatio.value = "0.75";
    } else if (preset <= 1000) {
      semanticRatio.value = "0.65";
    } else {
      semanticRatio.value = "0.55";
    }
    await applyLinkProfile();
  };
}

const runBatchBtn = document.getElementById("runBatch");
const batchStatus = document.getElementById("batchStatus");
const batchResults = document.getElementById("batchResults");
const batchBody = document.getElementById("batchBody");
const batchFoot = document.getElementById("batchFoot");

function renderBatchResults(data) {
  const results = (data.result && data.result.results) || [];
  const summary = (data.result && data.result.summary) || {};
  const config = (data.result && data.result.config) || {};

  batchBody.innerHTML = "";
  for (const r of results) {
    const tr = document.createElement("tr");
    if (r.error) {
      tr.innerHTML = `<td>${r.sequence}</td><td colspan="9" style="color:#f87171">${r.error}</td>`;
    } else {
      tr.innerHTML = [
        `<td>${r.sequence}</td>`,
        `<td>${r.frames}</td>`,
        `<td>${r.unique_tracks}</td>`,
        `<td>${r.semantic_kbps}</td>`,
        `<td>${r.baseline_kbps}</td>`,
        `<td>${r.bandwidth_savings_pct}%</td>`,
        `<td>${r.semantic_drop_pct}%</td>`,
        `<td>${r.baseline_drop_pct}%</td>`,
        `<td>${r.power_proxy_semantic_mbits} Mb</td>`,
        `<td>${r.power_proxy_baseline_mbits} Mb</td>`,
      ].join("");
    }
    batchBody.appendChild(tr);
  }

  batchFoot.innerHTML = "";
  if (summary.sequences_evaluated > 0) {
    const tr = document.createElement("tr");
    tr.style.fontWeight = "bold";
    tr.innerHTML = [
      `<td>AVG (${summary.sequences_evaluated} seqs)</td>`,
      `<td>${summary.total_frames}</td>`,
      `<td>${summary.total_unique_tracks}</td>`,
      `<td>${summary.avg_semantic_kbps}</td>`,
      `<td>${summary.avg_baseline_kbps}</td>`,
      `<td>${summary.avg_bandwidth_savings_pct}%</td>`,
      `<td>${summary.avg_semantic_drop_pct}%</td>`,
      `<td>${summary.avg_baseline_drop_pct}%</td>`,
      `<td>${summary.total_power_proxy_semantic_mbits} Mb</td>`,
      `<td>${summary.total_power_proxy_baseline_mbits} Mb (saved ${summary.power_savings_pct}%)</td>`,
    ].join("");
    batchFoot.appendChild(tr);
  }

  const cfgStr = config.model_path
    ? `model=${config.model_path} tracker=${config.tracker} link=${config.link_kbps}kbps loss=${config.packet_loss}`
    : "";
  batchStatus.textContent = `Done. ${cfgStr}`;
  batchResults.style.display = "";
}

async function pollBatchStatus() {
  try {
    const res = await fetch("/api/batch_status", { cache: "no-store" });
    const data = await res.json();
    if (data.error) {
      batchStatus.textContent = `Error: ${data.error}`;
      runBatchBtn.disabled = false;
      clearInterval(batchPollTimer);
      batchPollTimer = null;
      return;
    }
    if (!data.running && data.result) {
      clearInterval(batchPollTimer);
      batchPollTimer = null;
      runBatchBtn.disabled = false;
      renderBatchResults(data);
      return;
    }
    batchStatus.textContent = "Running\u2026";
  } catch (err) {
    batchStatus.textContent = `Poll failed: ${err}`;
  }
}

async function startBatch() {
  const dir = document.getElementById("batchDatasetDir").value.trim();
  if (!dir) {
    batchStatus.textContent = "Dataset directory is required.";
    return;
  }
  runBatchBtn.disabled = true;
  batchResults.style.display = "none";
  batchStatus.textContent = "Starting\u2026";

  const payload = {
    dataset_dir: dir,
    model_path: resolvedModelPath(),
    tracker: trackingTracker.value || "bytetrack.yaml",
    conf: 0.25,
    link_kbps: Number(document.getElementById("batchLinkKbps").value) || 1000,
    packet_loss: Number(document.getElementById("batchPacketLoss").value) || 0.02,
    semantic_ratio: Number(document.getElementById("semanticRatio").value) || 0.65,
    max_sequences: Number(document.getElementById("batchMaxSeqs").value) || 5,
  };

  try {
    const res = await fetch("/api/batch_run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      batchStatus.textContent = `Failed to start: ${data.error || res.status}`;
      runBatchBtn.disabled = false;
      return;
    }
    batchPollTimer = setInterval(pollBatchStatus, 2000);
  } catch (err) {
    batchStatus.textContent = `Request failed: ${err}`;
    runBatchBtn.disabled = false;
  }
}

if (trackingModelSelect) {
  trackingModelSelect.onchange = () => {
    trackingModelSelect._userTouched = true;
    trackingModelCustomRow.style.display = trackingModelSelect.value === "custom" ? "" : "none";
  };
}
if (trackingTracker) {
  trackingTracker.onchange = () => {
    trackingTracker._userTouched = true;
  };
}

applyLink.onclick = applyLinkProfile;
triggerRFSilence.onclick = triggerRFSilenceWindow;
applyVideoSourceBtn.onclick = applyVideoSource;
useSyntheticSourceBtn.onclick = useSyntheticSource;
applyTrackingConfigBtn.onclick = applyTrackingConfig;
requestKeyframe.onclick = sendKeyframeRequest;
refreshSummary.onclick = loadSummary;
if (runBatchBtn) runBatchBtn.onclick = startBatch;

function renderLoop() {
  renderSemanticFrame();
  requestAnimationFrame(renderLoop);
}

drawPlaceholder();
setInterval(pollState, 500);
setInterval(loadSummary, 4000);
pollState();
loadSummary();
requestAnimationFrame(renderLoop);
