const STORAGE_KEY = "alpasim.web_debugger.session_id";

const state = {
  sessionId: null,
  session: null,
  scenes: [],
  snapshotHistory: [],
  viewedTickId: null,
  refreshInFlight: false,
};

const els = {
  errorBanner: document.getElementById("error-banner"),
  headerScene: document.getElementById("header-scene"),
  headerTick: document.getElementById("header-tick"),
  headerStatus: document.getElementById("header-status"),
  mainPageLink: document.getElementById("main-page-link"),
  refresh: document.getElementById("refresh"),
  pause: document.getElementById("pause"),
  resume: document.getElementById("resume"),
  step: document.getElementById("step"),
  reconnect: document.getElementById("reconnect"),
  prevTick: document.getElementById("prev-tick"),
  nextTick: document.getElementById("next-tick"),
  sceneId: document.getElementById("scene-id"),
  createSession: document.getElementById("create-session"),
  applyBackends: document.getElementById("apply-backends"),
  backendSelector: document.getElementById("backend-selector"),
  timelineList: document.getElementById("timeline-list"),
  selectedBackend: document.getElementById("selected-backend"),
  selectedModel: document.getElementById("selected-model"),
  fallbackReason: document.getElementById("fallback-reason"),
  proposalCount: document.getElementById("proposal-count"),
  arbitrationReason: document.getElementById("arbitration-reason"),
  selectedCandidate: document.getElementById("selected-candidate"),
  snapshotTime: document.getElementById("snapshot-time"),
  snapshotInput: document.getElementById("snapshot-input"),
  candidateList: document.getElementById("candidate-list"),
  timingMetrics: document.getElementById("timing-metrics"),
  qualityMetrics: document.getElementById("quality-metrics"),
  selectedDebugJson: document.getElementById("selected-debug-json"),
};

function showError(message) {
  els.errorBanner.textContent = message;
  els.errorBanner.classList.remove("hidden");
}

function clearError() {
  els.errorBanner.textContent = "";
  els.errorBanner.classList.add("hidden");
}

async function readResponse(response) {
  if (response.ok) return response.json();
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

function loadSavedSessionId() {
  return window.localStorage.getItem(STORAGE_KEY) || "";
}

function saveSessionId(sessionId) {
  if (sessionId) window.localStorage.setItem(STORAGE_KEY, sessionId);
}

function activeSnapshot() {
  if (state.viewedTickId === null) return state.snapshotHistory.at(-1) || state.session?.latest_snapshot || null;
  return state.snapshotHistory.find((snapshot) => snapshot.tick_id === state.viewedTickId) || state.snapshotHistory.at(-1) || null;
}

function formatTime(us) {
  return `${(us / 1e6).toFixed(2)}s`;
}

function pretty(value) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

function backendTone(backendId) {
  const key = String(backendId || "").toLowerCase();
  if (key.includes("pdm")) return { border: "border-rose-300", bg: "bg-rose-50", text: "text-rose-700" };
  if (key.includes("vam") || key.includes("vla")) return { border: "border-emerald-300", bg: "bg-emerald-50", text: "text-emerald-700" };
  return { border: "border-violet-300", bg: "bg-violet-50", text: "text-violet-700" };
}

function selectedCandidate(snapshot) {
  const decision = snapshot?.latest_decision || state.session?.latest_decision;
  if (!decision) return null;
  return (decision.candidates || []).find((candidate) => candidate.selected) || null;
}

function flattenEntries(payload, prefix = "") {
  if (!payload || typeof payload !== "object") return [];
  return Object.entries(payload).flatMap(([key, value]) => {
    const nextKey = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return flattenEntries(value, nextKey);
    }
    return [[nextKey, value]];
  });
}

function renderMetricTable(container, payload) {
  if (!container) return;
  container.innerHTML = "";
  const rows = flattenEntries(payload);
  if (!rows.length) {
    container.innerHTML = '<div class="text-[11px] text-slate-400">暂无数据</div>';
    return;
  }
  rows.forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "kv-row";
    row.innerHTML = `<span class="text-slate-500">${key}</span><span class="font-mono font-semibold text-slate-900">${pretty(value)}</span>`;
    container.appendChild(row);
  });
}

function renderSceneOptions() {
  els.sceneId.innerHTML = "";
  const scenes = state.scenes || [];
  if (!scenes.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "无可用场景";
    els.sceneId.appendChild(option);
    return;
  }
  scenes.forEach((scene) => {
    const option = document.createElement("option");
    option.value = scene.scene_id;
    option.textContent = scene.label || scene.scene_id;
    els.sceneId.appendChild(option);
  });
  const preferredSceneId = new URLSearchParams(window.location.search).get("scene_id");
  els.sceneId.value = preferredSceneId || scenes[0].scene_id;
}

function renderTimeline(snapshot) {
  els.timelineList.innerHTML = "";
  const history = [...state.snapshotHistory].sort((a, b) => a.tick_id - b.tick_id);
  if (!history.length) {
    els.timelineList.innerHTML = '<div class="text-[11px] text-slate-400">暂无快照</div>';
    return;
  }
  history.slice().reverse().forEach((item) => {
    const decision = item.latest_decision || {};
    const selected = (decision.candidates || []).find((candidate) => candidate.selected);
    const isActive = snapshot?.tick_id === item.tick_id;
    const tone = backendTone(selected?.backend_id);
    const card = document.createElement("button");
    card.className = `w-full rounded-xl border p-3 text-left ${isActive ? "border-brand-500 bg-brand-50" : "border-slate-200 bg-white hover:bg-slate-50"}`;
    card.innerHTML = `
      <div class="flex items-center justify-between">
        <span class="font-mono text-[11px] font-bold text-slate-900">T${item.tick_id}</span>
        <span class="text-[10px] text-slate-500">${formatTime(item.sim_time_us)}</span>
      </div>
      <div class="mt-2 text-[11px] font-semibold ${selected ? tone.text : "text-slate-500"}">${selected?.backend_id || "-"}</div>
      <div class="mt-1 truncate text-[10px] text-slate-400">${decision.arbitration_reason || "-"}</div>
    `;
    card.onclick = () => {
      state.viewedTickId = item.tick_id;
      renderAll();
    };
    els.timelineList.appendChild(card);
  });
}

function renderBackendSelector() {
  const session = state.session;
  const decision = activeSnapshot()?.latest_decision || session?.latest_decision;
  const available = session?.available_backend_ids?.length
    ? session.available_backend_ids
    : [...new Set((decision?.candidates || []).map((candidate) => candidate.backend_id))];
  const active = new Set(session?.active_backend_ids || []);
  els.backendSelector.innerHTML = "";
  available.forEach((backendId) => {
    const tone = backendTone(backendId);
    const label = document.createElement("label");
    label.className = `flex items-center justify-between rounded-lg border px-3 py-2 text-[11px] font-semibold ${tone.border} ${tone.bg}`;
    label.innerHTML = `<span>${backendId}</span><input type="checkbox" value="${backendId}" class="h-4 w-4 accent-slate-900" ${active.has(backendId) ? "checked" : ""} />`;
    els.backendSelector.appendChild(label);
  });
}

function renderCandidates(snapshot) {
  const decision = snapshot?.latest_decision || state.session?.latest_decision;
  els.candidateList.innerHTML = "";
  (decision?.candidates || []).forEach((candidate) => {
    const tone = backendTone(candidate.backend_id);
    const debug = candidate.diagnostics?.driver_debug || {};
    const card = document.createElement("div");
    card.className = `rounded-2xl border p-4 ${candidate.selected ? `${tone.border} ${tone.bg}` : "border-slate-200 bg-white"}`;
    card.innerHTML = `
      <div class="flex items-start justify-between gap-3">
        <div>
          <div class="text-[12px] font-black uppercase tracking-wide ${tone.text}">${candidate.backend_id}</div>
          <div class="mt-1 text-[10px] text-slate-500">${candidate.status}</div>
        </div>
        <button class="candidate-select rounded-lg px-3 py-1.5 text-[10px] font-bold ${candidate.selected ? "bg-slate-200 text-slate-500" : "bg-slate-900 text-white"}" ${candidate.selected ? "disabled" : ""}>
          ${candidate.selected ? "当前使用" : "选用"}
        </button>
      </div>
      <div class="mt-3 space-y-2 text-[11px]">
        <div class="kv-row"><span class="text-slate-500">candidate_id</span><span class="truncate font-mono text-slate-900">${candidate.candidate_id}</span></div>
        <div class="kv-row"><span class="text-slate-500">selected_model_type</span><span class="font-mono text-slate-900">${pretty(debug.selected_model_type || candidate.diagnostics?.model_type_override)}</span></div>
        <div class="kv-row"><span class="text-slate-500">fallback_reason</span><span class="font-mono ${debug.fallback_reason ? "text-rose-600" : "text-emerald-600"}">${pretty(debug.fallback_reason)}</span></div>
        <div class="kv-row"><span class="text-slate-500">proposal_count</span><span class="font-mono text-slate-900">${pretty(debug.proposal_count)}</span></div>
        <div class="kv-row"><span class="text-slate-500">nearby_lane_count</span><span class="font-mono text-slate-900">${pretty(debug.nearby_lane_count)}</span></div>
        <div class="kv-row"><span class="text-slate-500">actor_count</span><span class="font-mono text-slate-900">${pretty(debug.actor_count)}</span></div>
      </div>
    `;
    const button = card.querySelector(".candidate-select");
    if (button && !candidate.selected) {
      button.onclick = () => mutateSession("/api/candidates/select", { candidate_id: candidate.candidate_id });
    }
    els.candidateList.appendChild(card);
  });
}

function renderHeader(snapshot) {
  const session = state.session;
  const selected = selectedCandidate(snapshot);
  const debug = selected?.diagnostics?.driver_debug || {};
  els.headerScene.textContent = session?.scene_id || "未连接会话";
  els.headerTick.textContent = snapshot ? `T${snapshot.tick_id}` : "T-";
  els.headerStatus.textContent = (session?.status || "OFFLINE").toLowerCase();
  const mainTarget = new URL("/", window.location.origin);
  if (state.sessionId) mainTarget.searchParams.set("session_id", state.sessionId);
  if (session?.scene_id) mainTarget.searchParams.set("scene_id", session.scene_id);
  els.mainPageLink.href = mainTarget.toString();

  els.selectedBackend.textContent = selected?.backend_id || "-";
  els.selectedModel.textContent = `model: ${pretty(debug.selected_model_type || selected?.diagnostics?.model_type_override)}`;
  els.fallbackReason.textContent = pretty(debug.fallback_reason);
  els.proposalCount.textContent = `proposals: ${pretty(debug.proposal_count)}`;
  els.arbitrationReason.textContent = snapshot?.latest_decision?.arbitration_reason || session?.latest_decision?.arbitration_reason || "-";
  els.selectedCandidate.textContent = `candidate: ${snapshot?.latest_decision?.selected_candidate_id || session?.latest_decision?.selected_candidate_id || "-"}`;
  els.snapshotTime.textContent = snapshot ? formatTime(snapshot.sim_time_us) : "-";
  els.snapshotInput.textContent = `input: ${snapshot?.latest_decision?.input_snapshot_id || "-"}`;
  els.selectedDebugJson.textContent = JSON.stringify(selected?.diagnostics || {}, null, 2);
}

function renderAll() {
  const snapshot = activeSnapshot();
  renderHeader(snapshot);
  renderBackendSelector();
  renderTimeline(snapshot);
  renderCandidates(snapshot);
  renderMetricTable(els.timingMetrics, snapshot?.context_diagnostics?.timing || {});
  renderMetricTable(els.qualityMetrics, snapshot?.context_diagnostics?.quality || {});
  if (window.lucide) window.lucide.createIcons();
}

async function loadScenes() {
  const payload = await api("/api/scenes");
  state.scenes = payload.scenes || [];
  renderSceneOptions();
}

async function refresh() {
  if (!state.sessionId || state.refreshInFlight) return;
  state.refreshInFlight = true;
  try {
    clearError();
    const session = await api(`/api/session/state?session_id=${encodeURIComponent(state.sessionId)}`);
    state.session = session;
    saveSessionId(state.sessionId);
    if (session.latest_snapshot) {
      const existing = state.snapshotHistory.findIndex((snapshot) => snapshot.tick_id === session.latest_snapshot.tick_id);
      if (existing >= 0) {
        state.snapshotHistory[existing] = session.latest_snapshot;
      } else {
        state.snapshotHistory.push(session.latest_snapshot);
        state.snapshotHistory.sort((a, b) => a.tick_id - b.tick_id);
      }
    }
    renderAll();
  } finally {
    state.refreshInFlight = false;
  }
}

async function createSession() {
  const sceneId = els.sceneId.value;
  const session = await api("/api/session/create", { scene_id: sceneId });
  state.sessionId = session.interactive_session_id;
  state.snapshotHistory = [];
  state.viewedTickId = null;
  await refresh();
}

async function reconnectSession(sessionId = "") {
  const target = sessionId || new URLSearchParams(window.location.search).get("session_id") || loadSavedSessionId();
  if (!target) return;
  state.sessionId = target;
  state.snapshotHistory = [];
  state.viewedTickId = null;
  await refresh();
}

async function mutateSession(path, body = {}) {
  if (!state.sessionId) return;
  clearError();
  await api(path, { session_id: state.sessionId, ...body });
  await refresh();
}

els.refresh.onclick = () => refresh().catch((error) => showError(error.message));
els.pause.onclick = () => mutateSession("/api/session/pause").catch((error) => showError(error.message));
els.resume.onclick = () => mutateSession("/api/session/resume").catch((error) => showError(error.message));
els.step.onclick = () => mutateSession("/api/session/step").catch((error) => showError(error.message));
els.reconnect.onclick = () => reconnectSession().catch((error) => showError(error.message));
els.createSession.onclick = () => createSession().catch((error) => showError(error.message));
els.applyBackends.onclick = () => {
  const backendIds = [...document.querySelectorAll("#backend-selector input:checked")].map((node) => node.value);
  mutateSession("/api/backends/active", { backend_ids: backendIds }).catch((error) => showError(error.message));
};
els.prevTick.onclick = () => {
  const snapshot = activeSnapshot();
  const history = state.snapshotHistory;
  const index = history.findIndex((item) => item.tick_id === snapshot?.tick_id);
  if (index > 0) {
    state.viewedTickId = history[index - 1].tick_id;
    renderAll();
  }
};
els.nextTick.onclick = () => {
  const snapshot = activeSnapshot();
  const history = state.snapshotHistory;
  const index = history.findIndex((item) => item.tick_id === snapshot?.tick_id);
  if (index >= 0 && index < history.length - 1) {
    state.viewedTickId = history[index + 1].tick_id;
    renderAll();
    return;
  }
  mutateSession("/api/session/step").catch((error) => showError(error.message));
};

Promise.all([loadScenes()])
  .then(() => reconnectSession())
  .catch((error) => showError(error.message));

setInterval(() => {
  if (state.session?.status === "RUNNING") {
    refresh().catch(() => {});
  }
}, 800);

renderAll();
lucide.createIcons();
