/**
 * Alpasim Webviz V2 - Core Application
 * Integrated Layout Management, Realistic Map Rendering, Session Persistence
 */

const state = {
  sessionId: null,
  sessionState: null,
  sensors: [],
  selectedSensorId: null,
  sceneMap: null,
  snapshotHistory: [],
  viewedTickId: null,
  refreshTimer: null,
  isEgoLocked: true,
  renderQueued: false,
};

const STORAGE_KEY = "alpasim_last_session_id";

const els = {
  sceneId: document.getElementById("scene-id"),
  createSession: document.getElementById("create-session"),
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
  statusIndicator: document.getElementById("status-indicator"),
  sessionStatus: document.getElementById("session-status"),
  sessionTick: document.getElementById("session-tick"),
  sessionTime: document.getElementById("session-time"),
  egoSpeed: document.getElementById("ego-speed"),
  sceneLabel: document.getElementById("scene-label"),
  mapScale: document.getElementById("map-scale"),
  mapSource: document.getElementById("map-source"),
  sensorTabs: document.getElementById("sensor-tabs"),
  primarySensorImage: document.getElementById("primary-sensor-image"),
  mapCanvas: document.getElementById("map-canvas"),
  chartCanvas: document.getElementById("chart-canvas"),
  candidateList: document.getElementById("candidate-list"),
  checkpointList: document.getElementById("checkpoint-list"),
  lockEgo: document.getElementById("lock-ego"),
  sessionsContainer: document.getElementById("sessions-container"),
  listSessions: document.getElementById("list-sessions"),
  reconnectLast: document.getElementById("reconnect-last"),
};

// --- API Helpers ---
async function api(path, body = null) {
  const r = await fetch(path, body ? { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) } : {});
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.grpc_details || err.error || `${r.status} ${r.statusText}`);
  }
  return r.json();
}

// --- Session Persistence & Management ---
async function saveSession(sid) {
  state.sessionId = sid;
  localStorage.setItem(STORAGE_KEY, sid);
  updateReconnectUI();
  await listSessions();
}

function updateReconnectUI() {
  const lastId = localStorage.getItem(STORAGE_KEY);
  if (lastId && lastId !== state.sessionId) {
    els.reconnectLast.textContent = `重连上次: ${lastId.slice(0, 8)}...`;
    els.reconnectLast.classList.remove("hidden");
  } else {
    els.reconnectLast.classList.add("hidden");
  }
}

async function attachSession(sid) {
  if (!sid) return;
  try {
    const sess = await api(`/api/session/state?session_id=${encodeURIComponent(sid)}`);
    // 如果获取状态成功，说明会话存在
    state.sessionId = sid;
    state.sessionState = sess;
    localStorage.setItem(STORAGE_KEY, sid);
    
    // 初始化该会话的数据
    const [sensors, map] = await Promise.all([
      api(`/api/sensors?session_id=${encodeURIComponent(sid)}`),
      api(`/api/map?scene_id=${encodeURIComponent(sess.scene_id)}`)
    ]);
    state.sensors = sensors.sensors;
    state.sceneMap = map;
    state.selectedSensorId = sensors.sensors[0]?.sensor_id;
    state.snapshotHistory = sess.latest_snapshot ? [sess.latest_snapshot] : [];
    
    await refresh(); // 开始同步循环
    console.log(`Successfully attached to session: ${sid}`);
  } catch (e) {
    console.warn(`Could not attach to session ${sid}:`, e.message);
    if (sid === state.sessionId) state.sessionId = null;
    updateReconnectUI();
  }
}

async function listSessions() {
  try {
    const data = await api("/api/sessions");
    els.sessionsContainer.innerHTML = "";
    if (data.session_ids && data.session_ids.length > 0) {
      data.session_ids.forEach(sid => {
        const isActive = sid === state.sessionId;
        const item = document.createElement("div");
        item.className = `session-item flex items-center justify-between p-2 rounded-lg border transition-all ${isActive ? 'bg-brand-50 border-brand-200' : 'bg-slate-50 border-slate-100 hover:border-slate-200'} group`;
        item.innerHTML = `
          <div class="flex flex-col min-w-0">
            <span class="text-[10px] font-bold truncate text-slate-700">${sid}</span>
            <span class="text-[8px] ${isActive ? 'text-brand-600 font-black' : 'text-slate-400'} uppercase tracking-tighter">
              ${isActive ? '● 当前活跃' : '活跃会话'}
            </span>
          </div>
          <button class="attach-btn ${isActive ? 'hidden' : 'opacity-0 group-hover:opacity-100'} text-[9px] font-black bg-white border border-slate-200 px-2 py-1 rounded shadow-sm hover:bg-brand-600 hover:text-white hover:border-brand-600 transition-all" data-sid="${sid}">挂载</button>
        `;
        const btn = item.querySelector(".attach-btn");
        if (btn) btn.onclick = () => attachSession(sid);
        els.sessionsContainer.appendChild(item);
      });
    } else {
      els.sessionsContainer.innerHTML = '<div class="text-[10px] text-slate-400 italic text-center py-2">无活跃会话记录</div>';
    }
  } catch (e) {
    console.error("List sessions failed", e);
  }
}

// --- Layout & Map Engine ---
class MapEngine {
  constructor(canvas) {
    this.canvas = canvas; this.ctx = canvas.getContext("2d");
    this.cam = { x: 0, y: 0, zoom: 15 };
    this.initEvents();
  }
  initEvents() {
    this.canvas.onwheel = (e) => { e.preventDefault(); this.cam.zoom = Math.min(150, Math.max(1, this.cam.zoom * (e.deltaY > 0 ? 0.9 : 1.1))); state.isEgoLocked = false; scheduleRender(); };
  }
  render(snapshot, map) {
    const { ctx, canvas, cam } = this;
    syncCanvasRes(canvas);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state.isEgoLocked && snapshot) { cam.x = snapshot.ego.x; cam.y = snapshot.ego.y; }
    
    // Grid
    ctx.strokeStyle = "#f1f5f9"; ctx.lineWidth = 0.5;
    const step = 50;
    for(let i=0; i<canvas.width; i+=step) { ctx.beginPath(); ctx.moveTo(i,0); ctx.lineTo(i, canvas.height); ctx.stroke(); }
    for(let j=0; j<canvas.height; j+=step) { ctx.beginPath(); ctx.moveTo(0,j); ctx.lineTo(canvas.width, j); ctx.stroke(); }
    
    if (!snapshot) return;
    
    // Draw Ego Body (Physically accurate ~4.8m x 2.0m)
    const p = { x: canvas.width/2 + (snapshot.ego.x - cam.x)*cam.zoom, y: canvas.height/2 - (snapshot.ego.y - cam.y)*cam.zoom };
    ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(-(snapshot.ego.heading_deg * Math.PI/180 - Math.PI/2));
    ctx.fillStyle = "#0ea5e9"; ctx.beginPath(); ctx.roundRect(-1.0*cam.zoom, -2.4*cam.zoom, 2.0*cam.zoom, 4.8*cam.zoom, 4); ctx.fill();
    // Headlights
    ctx.fillStyle = "#fff"; ctx.fillRect(-0.8*cam.zoom, -2.3*cam.zoom, 0.4*cam.zoom, 0.2*cam.zoom); ctx.fillRect(0.4*cam.zoom, -2.3*cam.zoom, 0.4*cam.zoom, 0.2*cam.zoom);
    ctx.restore();
    
    els.mapScale.textContent = `${(50/cam.zoom).toFixed(1)}米`;
  }
  zoomIn() { this.cam.zoom *= 1.2; scheduleRender(); }
  zoomOut() { this.cam.zoom /= 1.2; scheduleRender(); }
}

// --- Main Loop & Sync ---
function renderAll() {
  const sess = state.sessionState;
  const snapshot = activeSnapshot();
  const status = sess?.status ?? "OFFLINE";

  els.statusIndicator.className = `w-2 h-2 rounded-full mr-2 ${status === 'RUNNING' ? 'bg-emerald-500 animate-pulse' : 'bg-slate-300'}`;
  els.sessionStatus.textContent = status === "RUNNING" ? "运行中" : status === "PAUSED" ? "已暂停" : status === "OFFLINE" ? "离线" : status;
  els.sessionTick.textContent = snapshot?.tick_id ?? sess?.current_tick_id ?? "-";
  els.sessionTime.textContent = snapshot ? `仿真时间: ${(snapshot.sim_time_us/1e6).toFixed(2)}秒` : "仿真时间: -";
  els.egoSpeed.textContent = snapshot ? (snapshot.ego.speed_mps * 3.6).toFixed(1) + " km/h" : "-";
  els.sceneLabel.textContent = sess?.scene_id ?? "未挂载会话";
  els.mapSource.textContent = `地图源: ${state.sceneMap?.source ?? "无"}`;

  window.map.render(snapshot, state.sceneMap);

  const hist = state.snapshotHistory;
  if (hist.length) {
    const active = snapshot ?? hist.at(-1);
    els.timelineSlider.min = hist[0].tick_id; els.timelineSlider.max = hist.at(-1).tick_id; els.timelineSlider.value = active.tick_id;
    els.timelineCurrent.textContent = `帧 T${active.tick_id}`; els.timelineEnd.textContent = `T${hist.at(-1).tick_id}`;
  }

  if (state.sessionId && snapshot && state.selectedSensorId) {
    const url = `/api/frame?session_id=${encodeURIComponent(state.sessionId)}&sensor_id=${encodeURIComponent(state.selectedSensorId)}&tick_id=${snapshot.tick_id}`;
    if (els.primarySensorImage.src !== location.origin + url) els.primarySensorImage.src = url;
  } else {
    els.primarySensorImage.removeAttribute("src");
  }

  // Lists
  els.checkpointList.innerHTML = state.checkpoints.slice(-5).reverse().map(ck => `
    <div class="flex items-center justify-between p-2 bg-slate-50 border border-slate-100 rounded-lg cursor-pointer hover:bg-slate-100 transition-all group" onclick="mutate('/api/checkpoints/restore', {checkpoint_id:'${ck.checkpoint_id}'})">
      <span class="text-[10px] font-bold text-slate-600 uppercase">Tick T${ck.tick_id}</span>
      <i data-lucide="corner-down-left" class="w-3 h-3 text-slate-300 group-hover:text-brand-500"></i>
    </div>
  `).join("");

  if (window.lucide) window.lucide.createIcons();
}

function activeSnapshot() {
  if (state.viewedTickId === null) return state.snapshotHistory.at(-1);
  return state.snapshotHistory.find(s => s.tick_id === state.viewedTickId) || state.snapshotHistory.at(-1);
}

function scheduleRender() {
  if (state.renderQueued) return;
  state.renderQueued = true;
  requestAnimationFrame(() => { state.renderQueued = false; renderAll(); });
}

async function refresh() {
  if (!state.sessionId) return;
  try {
    const sess = await api(`/api/session/state?session_id=${encodeURIComponent(state.sessionId)}`);
    state.sessionState = sess;
    const [map, cks] = await Promise.all([
      api(`/api/map?scene_id=${encodeURIComponent(sess.scene_id)}`),
      api(`/api/checkpoints?session_id=${encodeURIComponent(state.sessionId)}`)
    ]);
    state.sceneMap = map; state.checkpoints = cks.checkpoints || [];
    const snap = sess.latest_snapshot;
    if (snap && !state.snapshotHistory.some(s => s.tick_id === snap.tick_id)) {
      state.snapshotHistory.push(snap); state.snapshotHistory.sort((a,b) => a.tick_id - b.tick_id);
    }
    scheduleRender();
    updateReconnectUI();
  } catch (e) {
    if (e.message.includes("404") || e.message.includes("not found")) {
      console.warn("Session expired on backend");
      state.sessionId = null;
    }
    scheduleRender();
  }
}

async function mutate(path, body = {}) {
  if (!state.sessionId) return;
  await api(path, { session_id: state.sessionId, ...body });
  await refresh();
}

function syncCanvasRes(c) {
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) {
    c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
  }
}

// --- Initialization ---
window.map = new MapEngine(els.mapCanvas);

els.createSession.onclick = async () => {
  const sess = await api("/api/session/create", { scene_id: els.sceneId.value });
  await saveSession(sess.interactive_session_id);
  state.snapshotHistory = [];
  await attachSession(sess.interactive_session_id);
};

els.reconnectLast.onclick = () => {
  const lastId = localStorage.getItem(STORAGE_KEY);
  if (lastId) attachSession(lastId);
};

els.listSessions.onclick = listSessions;
els.playSession.onclick = () => mutate("/api/session/resume");
els.pauseSession.onclick = () => mutate("/api/session/pause");
els.nextFrame.onclick = async () => {
  const cur = activeSnapshot(); const latest = state.snapshotHistory.at(-1);
  if (cur && latest && cur.tick_id < latest.tick_id) {
    state.viewedTickId = state.snapshotHistory[state.snapshotHistory.findIndex(s => s.tick_id === cur.tick_id) + 1].tick_id;
    renderAll();
  } else { await api("/api/session/step", { session_id: state.sessionId }); await refresh(); }
};
els.prevFrame.onclick = () => {
  const cur = activeSnapshot(); const idx = state.snapshotHistory.findIndex(s => s.tick_id === cur?.tick_id);
  if (idx > 0) { state.viewedTickId = state.snapshotHistory[idx-1].tick_id; renderAll(); }
};
els.liveJump.onclick = () => { state.viewedTickId = null; scheduleRender(); };
els.timelineSlider.oninput = () => { state.viewedTickId = Number(els.timelineSlider.value); renderAll(); };

// Boot logic
(async () => {
  updateReconnectUI();
  const lastId = localStorage.getItem(STORAGE_KEY);
  if (lastId) {
    console.log("Auto-reconnecting to last session:", lastId);
    await attachSession(lastId);
  }
  await listSessions();
  setInterval(() => { if (state.sessionState?.status === 'RUNNING') refresh(); }, 1000);
  setInterval(listSessions, 5000); // 周期性刷新列表
})();

window.onresize = scheduleRender;
lucide.createIcons();
