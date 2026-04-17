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
  mapView: {
    zoom: 1,
    panX: 0,
    panY: 0,
    rotation: 0,
    hasCustomView: false,
    isDragging: false,
    dragMode: null,
    lastPointerX: 0,
    lastPointerY: 0,
  },
  sensorOverlay: {
    offsetX: 0,
    offsetY: 0,
    dragging: false,
    lastPointerX: 0,
    lastPointerY: 0,
  },
};

const MAP_STYLE = {
  background: "#f5f7fb",
  roadEdgeOutline: "#ffffff",
  roadEdge: "#b8c4d4",
  laneBoundaryOutline: "#ffffff",
  laneBoundary: "#d5deea",
  laneCenterOutline: "#ffffff",
  laneCenter: "#8ea2b8",
  laneCenterDash: "#6f879f",
  stopLineOutline: "#fff7ed",
  stopLine: "#f97316",
  otherLineOutline: "#f5f3ff",
  otherLine: "#a855f7",
  egoFill: "#0ea5e9",
  egoStroke: "#0369a1",
  actorFill: "#f59e0b",
  actorStroke: "#b45309",
  egoHistory: "#2563eb",
  egoHistoryGlow: "rgba(37, 99, 235, 0.18)",
  selectedPlan: "#10b981",
  selectedPlanGlow: "rgba(16, 185, 129, 0.18)",
  backendPlans: {
    vam: { stroke: "#10b981", glow: "rgba(16, 185, 129, 0.18)" },
    pdm: { stroke: "#ef4444", glow: "rgba(239, 68, 68, 0.18)" },
    default: { stroke: "#a855f7", glow: "rgba(168, 85, 247, 0.18)" },
  },
  grid: "#e9eef5",
};

const CHART_STYLE = {
  axis: "#94a3b8",
  grid: "#e2e8f0",
  speed: "#0ea5e9",
  speedFill: "rgba(14, 165, 233, 0.12)",
  steering: "#f97316",
  steeringFill: "rgba(249, 115, 22, 0.12)",
  text: "#475569",
  mutedText: "#94a3b8",
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
  sensorOverlay: document.getElementById("sensor-overlay"),
  sensorOverlayHeader: document.getElementById("sensor-overlay-header"),
  mapCanvas: document.getElementById("map-canvas"),
  chartCanvas: document.getElementById("chart-canvas"),
  steeringChartCanvas: document.getElementById("steering-chart-canvas"),
  speedCurrent: document.getElementById("speed-current"),
  steeringCurrent: document.getElementById("steering-current"),
  backendSelector: document.getElementById("backend-selector"),
  applyBackends: document.getElementById("apply-backends"),
  decisionReason: document.getElementById("decision-reason"),
  candidateList: document.getElementById("candidate-list"),
  checkpointList: document.getElementById("checkpoint-list"),
  sessionList: document.getElementById("session-list"),
  resetMapView: document.getElementById("reset-map-view"),
  mapCompassNeedle: document.getElementById("map-compass-needle"),
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

function formatSteering(rad) {
  return `${((rad || 0) * 180 / Math.PI).toFixed(2)} deg`;
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
  ctx.fillStyle = MAP_STYLE.background;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!snapshot) return;

  const projection = createProjection(canvas, snapshot, state.sceneMap);
  drawMapGrid(ctx, canvas, projection);
  drawMapLayers(ctx, projection, state.sceneMap?.layers || {});
  drawCandidatePlans(ctx, projection, snapshot.candidate_plans || []);
  drawTrajectory(
    ctx,
    projection,
    buildHistoryPath(snapshot),
    {
      strokeStyle: MAP_STYLE.egoHistory,
      glowStyle: MAP_STYLE.egoHistoryGlow,
      lineWidth: 3.2,
      glowWidth: 8.0,
    },
  );
  drawActor(ctx, projection, snapshot.ego, {
    fillStyle: MAP_STYLE.egoFill,
    strokeStyle: MAP_STYLE.egoStroke,
    lengthM: 4.8,
    widthM: 2.0,
  });
  (snapshot.actors || []).forEach((actor) =>
    drawActor(ctx, projection, actor, {
      fillStyle: MAP_STYLE.actorFill,
      strokeStyle: MAP_STYLE.actorStroke,
      lengthM: 4.5,
      widthM: 1.8,
    }),
  );
  positionSensorOverlay();
}

function createProjection(canvas, snapshot, sceneMap) {
  const bounds = sceneMap?.bounds;
  const mapView = state.mapView;
  if (bounds) {
    const width = Math.max(bounds.max_x - bounds.min_x, 1);
    const height = Math.max(bounds.max_y - bounds.min_y, 1);
    const padding = 24;
    const baseScale = Math.min(
      (canvas.width - padding * 2) / width,
      (canvas.height - padding * 2) / height,
    );
    const scale = baseScale * mapView.zoom;
    const worldCenterX = (bounds.min_x + bounds.max_x) / 2;
    const worldCenterY = (bounds.min_y + bounds.max_y) / 2;
    return {
      scale,
      projectPoint(x, y) {
        const rotated = rotateWorldPoint(
          x - worldCenterX,
          y - worldCenterY,
          mapView.rotation,
        );
        return {
          x: canvas.width / 2 + rotated.x * scale + mapView.panX,
          y: canvas.height / 2 - rotated.y * scale + mapView.panY,
        };
      },
    };
  }

  const centerX = snapshot.ego?.x ?? 0;
  const centerY = snapshot.ego?.y ?? 0;
  const scale = 8 * mapView.zoom;
  return {
    scale,
    projectPoint(x, y) {
      const rotated = rotateWorldPoint(
        x - centerX,
        y - centerY,
        mapView.rotation,
      );
      return {
        x: canvas.width / 2 + rotated.x * scale + mapView.panX,
        y: canvas.height / 2 - rotated.y * scale + mapView.panY,
      };
    },
  };
}

function rotateWorldPoint(x, y, rotation) {
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  return {
    x: x * cos - y * sin,
    y: x * sin + y * cos,
  };
}

function drawMapGrid(ctx, canvas, projection) {
  const gridSpacingPx = projection.scale * 10;
  if (!Number.isFinite(gridSpacingPx) || gridSpacingPx < 24) return;
  const spacing = Math.min(gridSpacingPx, 160);
  const offsetX = ((state.mapView.panX % spacing) + spacing) % spacing;
  const offsetY = ((state.mapView.panY % spacing) + spacing) % spacing;

  ctx.save();
  ctx.strokeStyle = MAP_STYLE.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = offsetX; x <= canvas.width; x += spacing) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
  }
  for (let y = offsetY; y <= canvas.height; y += spacing) {
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawMapLayers(ctx, projection, layers) {
  const drawPolylines = (
    polylines,
    primaryStrokeStyle,
    primaryLineWidth,
    dashed = false,
    outlineStrokeStyle = null,
    outlineLineWidth = null,
  ) => {
    if (!polylines?.length) return;
    polylines.forEach((polyline) => {
      if (!polyline?.length) return;
      const screenPoints = polyline.map(([x, y]) => projection.projectPoint(x, y));
      const strokePath = () => {
        ctx.beginPath();
        screenPoints.forEach((point, index) => {
          if (index === 0) {
            ctx.moveTo(point.x, point.y);
          } else {
            ctx.lineTo(point.x, point.y);
          }
        });
        ctx.stroke();
      };

      if (outlineStrokeStyle && outlineLineWidth) {
        ctx.strokeStyle = outlineStrokeStyle;
        ctx.lineWidth = outlineLineWidth;
        ctx.setLineDash([]);
        strokePath();
      }

      ctx.strokeStyle = primaryStrokeStyle;
      ctx.lineWidth = primaryLineWidth;
      ctx.setLineDash(dashed ? [10, 8] : []);
      strokePath();
    });
  };

  drawPolylines(
    layers.road_edge,
    MAP_STYLE.roadEdge,
    2.6,
    false,
    MAP_STYLE.roadEdgeOutline,
    4.8,
  );
  drawPolylines(
    layers.road_lane_left_edge,
    MAP_STYLE.laneBoundary,
    1.8,
    false,
    MAP_STYLE.laneBoundaryOutline,
    3.2,
  );
  drawPolylines(
    layers.road_lane_right_edge,
    MAP_STYLE.laneBoundary,
    1.8,
    false,
    MAP_STYLE.laneBoundaryOutline,
    3.2,
  );
  drawPolylines(
    layers.road_lane_center,
    MAP_STYLE.laneCenterDash,
    1.4,
    true,
    MAP_STYLE.laneCenterOutline,
    2.8,
  );
  drawPolylines(
    layers.stop_line,
    MAP_STYLE.stopLine,
    3.0,
    false,
    MAP_STYLE.stopLineOutline,
    5.4,
  );
  drawPolylines(
    layers.other_line,
    MAP_STYLE.otherLine,
    2.0,
    false,
    MAP_STYLE.otherLineOutline,
    3.8,
  );
  ctx.setLineDash([]);
}

function drawTrajectory(ctx, projection, points, style) {
  if (!points?.length || points.length < 2) return;
  const screenPoints = points.map((point) => projection.projectPoint(point.x, point.y));

  const strokePath = () => {
    ctx.beginPath();
    screenPoints.forEach((point, index) => {
      if (index === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
    ctx.stroke();
  };

  ctx.save();
  if (style.glowStyle && style.glowWidth) {
    ctx.strokeStyle = style.glowStyle;
    ctx.lineWidth = style.glowWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.setLineDash([]);
    strokePath();
  }
  ctx.strokeStyle = style.strokeStyle;
  ctx.lineWidth = style.lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash(style.dashed ? [10, 8] : []);
  strokePath();
  ctx.restore();
}

function planStyleForBackend(backendId, selected) {
  const key = String(backendId || "").toLowerCase();
  const base =
    key.includes("pdm")
      ? MAP_STYLE.backendPlans.pdm
      : key.includes("vam") || key.includes("vla")
        ? MAP_STYLE.backendPlans.vam
        : MAP_STYLE.backendPlans.default;
  return {
    strokeStyle: base.stroke,
    glowStyle: base.glow,
    lineWidth: selected ? 4.2 : 2.6,
    glowWidth: selected ? 10 : 6,
    dashed: !selected,
  };
}

function drawCandidatePlans(ctx, projection, candidatePlans) {
  candidatePlans.forEach((candidatePlan) => {
    drawTrajectory(
      ctx,
      projection,
      candidatePlan.points || [],
      planStyleForBackend(candidatePlan.backend_id, candidatePlan.selected),
    );
  });
}

function drawActor(ctx, projection, actor, style) {
  if (!actor) return;
  const headingRad = ((actor.heading_deg || 0) * Math.PI) / 180;
  const cos = Math.cos(headingRad);
  const sin = Math.sin(headingRad);
  const halfLength = style.lengthM / 2;
  const halfWidth = style.widthM / 2;
  const worldCorners = [
    { x: halfLength, y: halfWidth },
    { x: halfLength, y: -halfWidth },
    { x: -halfLength, y: -halfWidth },
    { x: -halfLength, y: halfWidth },
  ].map((point) => ({
    x: actor.x + point.x * cos - point.y * sin,
    y: actor.y + point.x * sin + point.y * cos,
  }));
  const corners = worldCorners.map((point) => projection.projectPoint(point.x, point.y));

  ctx.beginPath();
  corners.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  ctx.closePath();
  ctx.fillStyle = style.fillStyle;
  ctx.strokeStyle = style.strokeStyle;
  ctx.lineWidth = 1.5;
  ctx.fill();
  ctx.stroke();
}

function buildHistoryPath(snapshot) {
  if (snapshot?.ego_history?.length) {
    return snapshot.ego_history;
  }
  if (snapshot?.tick_id == null) return [];
  return state.snapshotHistory
    .filter((item) => item.tick_id <= snapshot.tick_id)
    .map((item) => ({ x: item.ego.x, y: item.ego.y }));
}

function defaultMapRotation(snapshot) {
  const egoHeadingRad = (((snapshot?.ego?.heading_deg || 0) * Math.PI) / 180);
  return Math.PI / 2 - egoHeadingRad;
}

function applyDefaultMapView(snapshot) {
  state.mapView.zoom = 1;
  state.mapView.panX = 0;
  state.mapView.panY = 0;
  state.mapView.rotation = defaultMapRotation(snapshot);
  state.mapView.hasCustomView = false;
  state.mapView.isDragging = false;
  state.mapView.dragMode = null;
}

function resetMapView(snapshot = activeSnapshot()) {
  applyDefaultMapView(snapshot);
  renderAll();
}

function attachMapInteractions() {
  const canvas = els.mapCanvas;
  if (!canvas || canvas.dataset.interactionsAttached === "true") return;
  canvas.dataset.interactionsAttached = "true";

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const zoomFactor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    state.mapView.zoom = Math.min(12, Math.max(0.2, state.mapView.zoom * zoomFactor));
    state.mapView.hasCustomView = true;
    renderAll();
  });

  canvas.addEventListener("mousedown", (event) => {
    event.preventDefault();
    state.mapView.isDragging = true;
    state.mapView.dragMode = event.shiftKey ? "rotate" : "pan";
    state.mapView.lastPointerX = event.clientX;
    state.mapView.lastPointerY = event.clientY;
  });

  window.addEventListener("mousemove", (event) => {
    if (!state.mapView.isDragging) return;
    const deltaX = event.clientX - state.mapView.lastPointerX;
    const deltaY = event.clientY - state.mapView.lastPointerY;
    state.mapView.lastPointerX = event.clientX;
    state.mapView.lastPointerY = event.clientY;

    if (state.mapView.dragMode === "rotate") {
      state.mapView.rotation += deltaX * 0.01;
    } else {
      state.mapView.panX += deltaX;
      state.mapView.panY += deltaY;
    }
    state.mapView.hasCustomView = true;
    renderAll();
  });

  const endDrag = () => {
    state.mapView.isDragging = false;
    state.mapView.dragMode = null;
  };
  window.addEventListener("mouseup", endDrag);
  canvas.addEventListener("mouseleave", endDrag);
}

function attachSensorOverlayInteractions() {
  const header = els.sensorOverlayHeader;
  const overlay = els.sensorOverlay;
  if (!header || !overlay || header.dataset.interactionsAttached === "true") return;
  header.dataset.interactionsAttached = "true";

  header.addEventListener("mousedown", (event) => {
    if (event.target.closest("button")) return;
    event.preventDefault();
    state.sensorOverlay.dragging = true;
    state.sensorOverlay.lastPointerX = event.clientX;
    state.sensorOverlay.lastPointerY = event.clientY;
  });

  window.addEventListener("mousemove", (event) => {
    if (!state.sensorOverlay.dragging) return;
    const deltaX = event.clientX - state.sensorOverlay.lastPointerX;
    const deltaY = event.clientY - state.sensorOverlay.lastPointerY;
    state.sensorOverlay.lastPointerX = event.clientX;
    state.sensorOverlay.lastPointerY = event.clientY;
    state.sensorOverlay.offsetX += deltaX;
    state.sensorOverlay.offsetY += deltaY;
    positionSensorOverlay();
  });

  window.addEventListener("mouseup", () => {
    state.sensorOverlay.dragging = false;
  });
}

function positionSensorOverlay() {
  const overlay = els.sensorOverlay;
  if (!overlay) return;
  overlay.style.transform = `translate(${state.sensorOverlay.offsetX}px, ${state.sensorOverlay.offsetY}px)`;
}

function renderChart() {
  const history = state.snapshotHistory.slice(-90);
  const snapshot = activeSnapshot();
  if (els.speedCurrent) {
    els.speedCurrent.textContent = snapshot ? formatSpeed(snapshot.ego.speed_mps) : "-";
  }
  if (els.steeringCurrent) {
    els.steeringCurrent.textContent = snapshot
      ? formatSteering(snapshot.ego.front_steering_angle_rad)
      : "-";
  }
  renderLineChart(els.chartCanvas, history, {
    yAccessor: (snapshot) => snapshot.ego.speed_mps * 3.6,
    yUnit: "km/h",
    strokeStyle: CHART_STYLE.speed,
    fillStyle: CHART_STYLE.speedFill,
    minSpan: 10,
    symmetric: false,
  });
  renderLineChart(els.steeringChartCanvas, history, {
    yAccessor: trueFrontSteeringDeg,
    yUnit: "deg",
    strokeStyle: CHART_STYLE.steering,
    fillStyle: CHART_STYLE.steeringFill,
    minSpan: 8,
    symmetric: true,
  });
}

function trueFrontSteeringDeg(snapshot) {
  return ((snapshot?.ego?.front_steering_angle_rad || 0) * 180) / Math.PI;
}

function renderLineChart(canvas, history, style) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(Math.floor(rect.width), 240);
  canvas.height = Math.max(Math.floor(rect.height), 160);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const margin = { top: 16, right: 16, bottom: 26, left: 46 };
  const plotWidth = Math.max(canvas.width - margin.left - margin.right, 10);
  const plotHeight = Math.max(canvas.height - margin.top - margin.bottom, 10);

  drawChartFrame(ctx, canvas, margin, plotWidth, plotHeight);
  if (history.length < 2) {
    drawChartEmptyState(ctx, canvas, margin);
    return;
  }

  const samples = history
    .map((snapshot) => ({
      t: snapshot.sim_time_us / 1e6,
      v: style.yAccessor(snapshot),
    }))
    .filter((sample) => Number.isFinite(sample.t) && Number.isFinite(sample.v));
  if (samples.length < 2) {
    drawChartEmptyState(ctx, canvas, margin);
    return;
  }
  const times = samples.map((sample) => sample.t);
  const values = samples.map((sample) => sample.v);
  const timeMin = times[0];
  const timeMax = times[times.length - 1];
  const timeSpan = Math.max(timeMax - timeMin, 1e-6);

  let yMin = Math.min(...values);
  let yMax = Math.max(...values);
  if (style.symmetric) {
    const bound = Math.max(Math.abs(yMin), Math.abs(yMax), style.minSpan / 2);
    yMin = -bound;
    yMax = bound;
  } else {
    yMin = Math.min(0, yMin);
    yMax = Math.max(yMax, style.minSpan);
    if (yMax - yMin < style.minSpan) {
      yMax = yMin + style.minSpan;
    }
  }
  const ySpan = Math.max(yMax - yMin, 1e-6);

  const xOf = (timeSec) => margin.left + ((timeSec - timeMin) / timeSpan) * plotWidth;
  const yOf = (value) => margin.top + (1 - (value - yMin) / ySpan) * plotHeight;

  drawChartGrid(ctx, margin, plotWidth, plotHeight, yMin, yMax, style);
  drawChartAxes(ctx, canvas, margin, plotWidth, plotHeight, yMin, yMax, timeMin, timeMax, style);
  drawChartSeries(ctx, samples, xOf, yOf, margin, plotHeight, style);
}

function drawChartFrame(ctx, canvas, margin, plotWidth, plotHeight) {
  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#f1f5f9";
  ctx.strokeRect(margin.left, margin.top, plotWidth, plotHeight);
  ctx.restore();
}

function drawChartEmptyState(ctx, canvas, margin) {
  ctx.save();
  ctx.fillStyle = CHART_STYLE.mutedText;
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText("至少需要两帧数据", margin.left + 8, canvas.height / 2);
  ctx.restore();
}

function drawChartGrid(ctx, margin, plotWidth, plotHeight, yMin, yMax, style) {
  ctx.save();
  ctx.strokeStyle = CHART_STYLE.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = margin.top + (i / 4) * plotHeight;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotWidth, y);
    ctx.stroke();
  }
  if (style.symmetric && yMin < 0 && yMax > 0) {
    const zeroY = margin.top + (1 - (0 - yMin) / (yMax - yMin)) * plotHeight;
    ctx.strokeStyle = "#cbd5e1";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(margin.left, zeroY);
    ctx.lineTo(margin.left + plotWidth, zeroY);
    ctx.stroke();
  }
  ctx.restore();
}

function drawChartAxes(ctx, canvas, margin, plotWidth, plotHeight, yMin, yMax, timeMin, timeMax, style) {
  ctx.save();
  ctx.fillStyle = CHART_STYLE.text;
  ctx.strokeStyle = CHART_STYLE.axis;
  ctx.lineWidth = 1;
  ctx.font = "10px Inter, sans-serif";

  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotHeight);
  ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  ctx.stroke();

  for (let i = 0; i <= 4; i += 1) {
    const value = yMax - ((yMax - yMin) * i) / 4;
    const y = margin.top + (i / 4) * plotHeight;
    ctx.fillText(`${value.toFixed(1)}`, 4, y + 3);
  }

  for (let i = 0; i <= 3; i += 1) {
    const timeValue = timeMin + ((timeMax - timeMin) * i) / 3;
    const x = margin.left + (i / 3) * plotWidth;
    ctx.fillText(`${timeValue.toFixed(1)}s`, x - 8, canvas.height - 8);
  }

  ctx.fillText(style.yUnit, 6, 12);
  ctx.fillText("time", canvas.width - 34, canvas.height - 8);
  ctx.restore();
}

function drawChartSeries(ctx, samples, xOf, yOf, margin, plotHeight, style) {
  const baselineY = margin.top + plotHeight;

  ctx.save();
  ctx.beginPath();
  samples.forEach((sample, index) => {
    const x = xOf(sample.t);
    const y = yOf(sample.v);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  const lastX = xOf(samples[samples.length - 1].t);
  const firstX = xOf(samples[0].t);
  ctx.lineTo(lastX, baselineY);
  ctx.lineTo(firstX, baselineY);
  ctx.closePath();
  ctx.fillStyle = style.fillStyle;
  ctx.fill();

  ctx.beginPath();
  samples.forEach((sample, index) => {
    const x = xOf(sample.t);
    const y = yOf(sample.v);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.strokeStyle = style.strokeStyle;
  ctx.lineWidth = 2.2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();
  ctx.restore();
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

  if (els.backendSelector) {
    const availableBackends = session?.available_backend_ids?.length
      ? session.available_backend_ids
      : [...new Set((decision?.candidates || []).map((candidate) => candidate.backend_id))];
    const activeBackends = new Set(session?.active_backend_ids || []);
    els.backendSelector.innerHTML = "";
    availableBackends.forEach((backendId) => {
      const label = document.createElement("label");
      label.className =
        "flex items-center justify-between rounded-md border border-slate-200 bg-white px-2 py-1 text-[10px] font-medium text-slate-600";
      label.innerHTML = `
        <span class="truncate">${backendId}</span>
        <input type="checkbox" value="${backendId}" class="h-3.5 w-3.5 accent-slate-900" ${
          activeBackends.has(backendId) ? "checked" : ""
        } />
      `;
      els.backendSelector.appendChild(label);
    });
  }

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
      <div class="mt-2 flex justify-end">
        <button class="candidate-select rounded-md px-2 py-1 text-[9px] font-bold ${
          candidate.selected
            ? "bg-slate-200 text-slate-500"
            : "bg-brand-600 text-white"
        }" ${candidate.selected ? "disabled" : ""}>
          ${candidate.selected ? "当前使用" : "选用"}
        </button>
      </div>
    `;
    const selectButton = item.querySelector(".candidate-select");
    if (selectButton && !candidate.selected) {
      selectButton.onclick = () =>
        mutateSession("/api/candidates/select", {
          candidate_id: candidate.candidate_id,
        }).catch((error) => showError(error.message));
    }
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
  if (els.mapCompassNeedle) {
    els.mapCompassNeedle.style.transform = `rotate(${-state.mapView.rotation}rad)`;
  }
  if (window.lucide) window.lucide.createIcons();
}

async function loadScenes() {
  const payload = await api("/api/scenes");
  state.availableScenes = payload.scenes || [];
  renderSceneOptions();
}

async function loadSceneMap(sceneId) {
  if (!sceneId) {
    state.sceneMap = null;
    return;
  }
  state.sceneMap = await api(`/api/map?scene_id=${encodeURIComponent(sceneId)}`);
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
    } else if (session.latest_snapshot) {
      state.snapshotHistory = state.snapshotHistory.map((snapshot) =>
        snapshot.tick_id === session.latest_snapshot.tick_id ? session.latest_snapshot : snapshot,
      );
    }
    if (session.latest_snapshot && !state.mapView.hasCustomView) {
      applyDefaultMapView(session.latest_snapshot);
    }
    if (!state.sceneMap || state.sceneMap.scene_id !== session.scene_id) {
      await loadSceneMap(session.scene_id);
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
  state.sceneMap = null;
  state.mapView.hasCustomView = false;
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
  state.sceneMap = null;
  state.mapView.hasCustomView = false;
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
els.resetMapView.onclick = () => resetMapView();
els.applyBackends.onclick = () => {
  const selected = [...document.querySelectorAll("#backend-selector input:checked")].map(
    (input) => input.value,
  );
  mutateSession("/api/backends/active", { backend_ids: selected }).catch((error) =>
    showError(error.message),
  );
};

attachMapInteractions();
attachSensorOverlayInteractions();

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
