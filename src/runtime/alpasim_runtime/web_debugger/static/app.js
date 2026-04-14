/**
 * Alpasim Standard Console - Core Logic
 */

const STORAGE_KEYS = {
  sessionId: "alpasim.web_debugger.session_id",
};

const state = {
  sessionId: null,
  sessionState: null,
  availableScenes: [],
  availableSessions: [],
  sensors: [],
  selectedSensorId: null,
  sceneMap: null,
  snapshotHistory: [],
  viewedTickId: null,
  refreshInFlight: false,
};

const els = {
  errorBanner: document.getElementById("error-banner"),
  sceneId: document.getElementById("scene-id"),
  createSession: document.getElementById("create-session"),
  reconnectSession: document.getElementById("reconnect-session"),
  refreshState: document.getElementById("refresh-state"),
  playSession: document.getElementById("play-session"),
  pauseSession: document.getElementById("pause-session"),
  prevFrame: document.getElementById("prev-frame"),
  nextFrame: document.getElementById("next-frame"),
  liveJump: document.getElementById("live-jump"),
  timelineSlider: document.getElementById("timeline-slider"),
  playbackMode: document.getElementById("playback-mode"),
  timelineCurrent: document.getElementById("timeline-current"),
  timelineEnd: document.getElementById("timeline-end"),
  sessionStatus: document.getElementById("session-status"),
  statusDot: document.getElementById("status-dot"),
  sessionTick: document.getElementById("session-tick"),
  sessionTime: document.getElementById("session-time"),
  egoSpeed: document.getElementById("ego-speed"),
  sceneLabel: document.getElementById("scene-label"),
  sensorTabs: document.getElementById("sensor-tabs"),
  primarySensorImage: document.getElementById("primary-sensor-image"),
  mapCanvas: document.getElementById("map-canvas"),
  chartCanvas: document.getElementById("chart-canvas"),
  decisionReason: document.getElementById("decision-reason"),
  candidateList: document.getElementById("candidate-list"),
  checkpointList: document.getElementById("checkpoint-list"),
  sessionList: document.getElementById("session-list"),
};

async function readResponse(response) {
  if (response.ok) {
    return response.json();
  }
  const payload = await response.json().catch(() => ({}));
  throw new Error(payload.grpc_details || payload.error || `${response.status} ${response.statusText}`);
}

async function api(path, body = null) {
  const response = await fetch(
    path,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : undefined,
  );
  return readResponse(response);
}

function showError(message) {
  if (!els.errorBanner) return;
  els.errorBanner.textContent = message;
  els.errorBanner.classList.remove("hidden");
}

function clearError() {
  if (!els.errorBanner) return;
  els.errorBanner.textContent = "";
  els.errorBanner.classList.add("hidden");
}

function saveSessionId(sessionId) {
  if (!sessionId) return;
  window.localStorage.setItem(STORAGE_KEYS.sessionId, sessionId);
}

function loadSavedSessionId() {
  return window.localStorage.getItem(STORAGE_KEYS.sessionId) || "";
}

function clearSavedSessionId() {
  window.localStorage.removeItem(STORAGE_KEYS.sessionId);
}

function activeSnapshot() {
  if (state.viewedTickId === null) return state.snapshotHistory.at(-1);
  return (
    state.snapshotHistory.find((snapshot) => snapshot.tick_id === state.viewedTickId) ||
    state.snapshotHistory.at(-1)
  );
}

function formatSimTime(us) {
  return `${(us / 1e6).toFixed(2)}s`;
}

function formatSpeed(mps) {
  return `${(mps * 3.6).toFixed(1)} km/h`;
}

function renderSceneOptions() {
  const currentValue = els.sceneId.value;
  els.sceneId.innerHTML = "";
  if (!state.availableScenes.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "无可用场景";
    els.sceneId.appendChild(option);
    return;
  }
  state.availableScenes.forEach((scene) => {
    const option = document.createElement("option");
    option.value = scene.scene_id;
    option.textContent = scene.label || scene.scene_id;
    els.sceneId.appendChild(option);
  });
  const preferredSceneId = new URLSearchParams(window.location.search).get("scene_id");
  els.sceneId.value = currentValue || preferredSceneId || state.availableScenes[0].scene_id;
}

function renderMap(snapshot) {
  const canvas = els.mapCanvas;
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!snapshot) return;

  ctx.fillStyle = "#0ea5e9";
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  ctx.beginPath();
  ctx.roundRect(centerX - 10, centerY - 20, 20, 40, 4);
  ctx.fill();
}

function renderChart() {
  const canvas = els.chartCanvas;
  const rect = canvas.parentNode.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const history = state.snapshotHistory.slice(-60);
  if (history.length < 2) return;

  const maxSpeed = Math.max(...history.map((snapshot) => snapshot.ego.speed_mps), 5);
  ctx.strokeStyle = "#0ea5e9";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((snapshot, index) => {
    const x = (index / (history.length - 1)) * canvas.width;
    const y = canvas.height - (snapshot.ego.speed_mps / maxSpeed) * canvas.height;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function renderSensorFrame(snapshot) {
  if (!state.sessionId || !snapshot || !state.selectedSensorId) {
    els.primarySensorImage.removeAttribute("src");
  } else {
    els.primarySensorImage.src =
      `/api/frame?session_id=${encodeURIComponent(state.sessionId)}` +
      `&sensor_id=${encodeURIComponent(state.selectedSensorId)}` +
      `&tick_id=${snapshot.tick_id}`;
  }

  els.sensorTabs.innerHTML = "";
  state.sensors.forEach((sensor) => {
    const button = document.createElement("button");
    button.className =
      "rounded-md px-2 py-1 text-[10px] font-bold " +
      (sensor.sensor_id === state.selectedSensorId
        ? "bg-brand-50 text-brand-600"
        : "bg-slate-100 text-slate-500");
    button.textContent = sensor.sensor_id.split("_")[1] || sensor.sensor_id;
    button.onclick = () => {
      state.selectedSensorId = sensor.sensor_id;
      renderAll();
    };
    els.sensorTabs.appendChild(button);
  });
}

function renderLists(session, snapshot) {
  const decision = snapshot?.latest_decision ?? session?.latest_decision;
  els.decisionReason.textContent = decision?.arbitration_reason ?? "等待数据...";

  els.candidateList.innerHTML = "";
  (decision?.candidates || []).forEach((candidate) => {
    const item = document.createElement("div");
    item.className =
      `p-2 border rounded-lg text-[10px] ${
        candidate.selected ? "border-brand-500 bg-brand-50" : "border-slate-100"
      }`;
    item.innerHTML = `
      <div class="flex justify-between font-bold">
        <span>${candidate.backend_id}</span>
        <span>${candidate.status}</span>
      </div>
      <div class="mt-1 truncate text-slate-400">${candidate.candidate_id}</div>
    `;
    els.candidateList.appendChild(item);
  });

  els.checkpointList.innerHTML = "";
  state.checkpoints.slice(-5).reverse().forEach((checkpoint) => {
    const item = document.createElement("div");
    item.className =
      "flex items-center justify-between p-2 bg-slate-50 rounded-lg border border-slate-100";
    item.innerHTML = `
      <span class="text-[10px] font-bold">T${checkpoint.tick_id}</span>
      <button class="text-[10px] font-bold text-brand-600">恢复</button>
    `;
    item.onclick = () =>
      mutateSession("/api/checkpoints/restore", {
        checkpoint_id: checkpoint.checkpoint_id,
      }).catch((error) => showError(error.message));
    els.checkpointList.appendChild(item);
  });
}

function renderSessionList() {
  if (!els.sessionList) return;
  els.sessionList.innerHTML = "";
  if (!state.availableSessions.length) {
    const empty = document.createElement("div");
    empty.className = "text-[10px] text-slate-400";
    empty.textContent = "无活跃会话";
    els.sessionList.appendChild(empty);
    return;
  }

  state.availableSessions.forEach((session) => {
    const item = document.createElement("div");
    const isCurrent = state.sessionId === session.interactive_session_id;
    item.className =
      "flex items-center justify-between rounded-lg border p-2 " +
      (isCurrent ? "border-brand-300 bg-brand-50" : "border-slate-100 bg-slate-50");
    item.innerHTML = `
      <div class="min-w-0">
        <div class="truncate text-[10px] font-bold text-slate-700">${session.scene_id}</div>
        <div class="truncate text-[9px] text-slate-400">${session.interactive_session_id}</div>
      </div>
      <button class="text-[10px] font-bold ${isCurrent ? "text-slate-400" : "text-brand-600"}">
        ${isCurrent ? "已连接" : "连接"}
      </button>
    `;
    if (!isCurrent) {
      item.querySelector("button").onclick = () =>
        reconnectSession(session.interactive_session_id).catch((error) => showError(error.message));
    } else {
      item.querySelector("button").disabled = true;
    }
    els.sessionList.appendChild(item);
  });
}

function renderAll() {
  const session = state.sessionState;
  const snapshot = activeSnapshot();
  const status = session?.status ?? "OFFLINE";

  els.statusDot.className =
    `w-2 h-2 rounded-full mr-2 ${
      status === "RUNNING" ? "bg-emerald-500 animate-pulse" : "bg-slate-300"
    }`;
  els.sessionStatus.textContent =
    status === "RUNNING" ? "运行中" : status === "PAUSED" ? "已暂停" : "离线";
  els.sessionTick.textContent = snapshot?.tick_id ?? "-";
  els.sessionTime.textContent = snapshot ? `时长: ${formatSimTime(snapshot.sim_time_us)}` : "时长: -";
  els.egoSpeed.textContent = snapshot ? formatSpeed(snapshot.ego.speed_mps) : "-";
  els.sceneLabel.textContent = session?.scene_id ?? "未加载场景";

  const history = state.snapshotHistory;
  if (history.length) {
    const active = snapshot ?? history.at(-1);
    els.timelineSlider.min = history[0].tick_id;
    els.timelineSlider.max = history.at(-1).tick_id;
    els.timelineSlider.value = active.tick_id;
    els.timelineCurrent.textContent = `帧 T${active.tick_id}`;
    els.timelineEnd.textContent = `T${history.at(-1).tick_id}`;
  }

  renderMap(snapshot);
  renderSensorFrame(snapshot);
  renderChart();
  renderLists(session, snapshot);
  renderSessionList();
  if (window.lucide) window.lucide.createIcons();
}

async function loadScenes() {
  const payload = await api("/api/scenes");
  state.availableScenes = payload.scenes || [];
  renderSceneOptions();
}

async function loadSessions() {
  const payload = await api("/api/sessions");
  state.availableSessions = payload.sessions || [];
  renderSessionList();
}

async function refresh() {
  if (!state.sessionId || state.refreshInFlight) return;
  state.refreshInFlight = true;
  try {
    clearError();
    const session = await api(`/api/session/state?session_id=${encodeURIComponent(state.sessionId)}`);
    state.sessionState = session;
    saveSessionId(state.sessionId);
    const [sensorsPayload, checkpointsPayload] = await Promise.all([
      api(`/api/sensors?session_id=${encodeURIComponent(state.sessionId)}`),
      api(`/api/checkpoints?session_id=${encodeURIComponent(state.sessionId)}`),
    ]);
    state.sensors = sensorsPayload.sensors || [];
    if (!state.selectedSensorId) {
      state.selectedSensorId = state.sensors[0]?.sensor_id || null;
    }
    state.checkpoints = checkpointsPayload.checkpoints || [];
    if (
      session.latest_snapshot &&
      !state.snapshotHistory.some((snapshot) => snapshot.tick_id === session.latest_snapshot.tick_id)
    ) {
      state.snapshotHistory.push(session.latest_snapshot);
      state.snapshotHistory.sort((left, right) => left.tick_id - right.tick_id);
    }
    renderAll();
  } catch (error) {
    if (String(error.message || "").includes("Unknown interactive_session_id")) {
      clearSavedSessionId();
      state.sessionId = null;
    }
    showError(error.message);
    throw error;
  } finally {
    state.refreshInFlight = false;
  }
}

async function createSession() {
  const sceneId = els.sceneId.value;
  if (!sceneId) {
    showError("请先选择场景。");
    return;
  }
  clearError();
  const session = await api("/api/session/create", { scene_id: sceneId });
  state.sessionId = session.interactive_session_id;
  saveSessionId(state.sessionId);
  state.snapshotHistory = [];
  state.viewedTickId = null;
  state.selectedSensorId = null;
  await refresh();
  await loadSessions();
}

async function reconnectSession(sessionId = "") {
  const targetSessionId =
    sessionId ||
    new URLSearchParams(window.location.search).get("session_id") ||
    loadSavedSessionId();
  if (!targetSessionId) {
    return;
  }
  clearError();
  state.sessionId = targetSessionId;
  state.snapshotHistory = [];
  state.viewedTickId = null;
  state.selectedSensorId = null;
  try {
    await refresh();
    await loadSessions();
  } catch (error) {
    clearSavedSessionId();
    throw error;
  }
}

async function mutateSession(path, body = {}) {
  if (!state.sessionId) {
    showError("当前没有已连接的会话。");
    return;
  }
  clearError();
  await api(path, { session_id: state.sessionId, ...body });
  await refresh();
  await loadSessions();
}

els.prevFrame.onclick = () => {
  const current = activeSnapshot();
  const index = state.snapshotHistory.findIndex((snapshot) => snapshot.tick_id === current?.tick_id);
  if (index > 0) {
    state.viewedTickId = state.snapshotHistory[index - 1].tick_id;
    renderAll();
  }
};

els.nextFrame.onclick = () => {
  const current = activeSnapshot();
  const latest = state.snapshotHistory.at(-1);
  if (current && latest && current.tick_id < latest.tick_id) {
    const index = state.snapshotHistory.findIndex((snapshot) => snapshot.tick_id === current.tick_id);
    state.viewedTickId = state.snapshotHistory[index + 1].tick_id;
    renderAll();
    return;
  }
  mutateSession("/api/session/step").catch((error) => showError(error.message));
};

els.createSession.onclick = () => createSession().catch((error) => showError(error.message));
els.reconnectSession.onclick = () => reconnectSession().catch((error) => showError(error.message));
els.refreshState.onclick = () => refresh().catch((error) => showError(error.message));
els.playSession.onclick = () => mutateSession("/api/session/resume").catch((error) => showError(error.message));
els.pauseSession.onclick = () => mutateSession("/api/session/pause").catch((error) => showError(error.message));
els.liveJump.onclick = () => {
  state.viewedTickId = null;
  renderAll();
};
els.timelineSlider.oninput = () => {
  state.viewedTickId = Number(els.timelineSlider.value);
  renderAll();
};

Promise.all([loadScenes(), loadSessions()])
  .then(() => reconnectSession())
  .catch((error) => showError(error.message));

setInterval(() => {
  if (state.sessionState?.status === "RUNNING") {
    refresh().catch(() => {});
  }
}, 800);

setInterval(() => {
  loadSessions().catch(() => {});
}, 2000);

renderAll();
lucide.createIcons();
