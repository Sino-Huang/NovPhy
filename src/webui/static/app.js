const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const emptyState = document.getElementById('emptyState');
const logEl = document.getElementById('log');
let selectedPoint = null;
let aimStartPoint = null;
let aimCurrentPoint = null;
let latestFrameImage = null;
let trajectoryWorldWidth = 17.5;
let trajectorySlingCenter = null;
let autoTimer = null;
let physicsReviewEnabled = false;
let pendingReviewAction = null;
let reviewSteps = [];

const WEB_TRAJECTORY_DRAG_RADIUS_WORLD = 1;
const WEB_TRAJECTORY_MAX_LAUNCH_SPEED = 10;
const WEB_TRAJECTORY_LAUNCH_GRAVITY = 0.48;
const WEB_TRAJECTORY_TIME_STEP = 0.02;
const WEB_TRAJECTORY_STEPS = 500;
const WEB_TRAJECTORY_CANVAS_Y_OFFSET = -1;

function log(message) {
  const stamp = new Date().toLocaleTimeString();
  logEl.textContent = `[${stamp}] ${message}\n` + logEl.textContent.split('\n').slice(0, 12).join('\n');
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function post(path, payload = {}) {
  return api(path, { method: 'POST', body: JSON.stringify(payload) });
}

function setStatus(status) {
  const el = document.getElementById('connectionStatus');
  const connected = Boolean(status.connected);
  el.textContent = connected ? 'Connected' : 'Disconnected';
  el.classList.toggle('connected', connected);
  physicsReviewEnabled = Boolean(status.physicsV2Review);
  document.getElementById('physicsReviewPanel').classList.toggle('hidden', !physicsReviewEnabled);
  const autoExecute = document.getElementById('autoExecuteAgentAction');
  if (physicsReviewEnabled) {
    autoExecute.checked = false;
    autoExecute.disabled = true;
  }
  if (status.physicsV2ReviewSession) updateReviewSession(status.physicsV2ReviewSession);
}

const reviewGuidance = {
  collision: {
    level: 1,
    text: 'Load public training level 1. Pull backward from the slingshot until the predicted arc crosses a visible block or pig; do not aim into empty sky.',
  },
  'persistent support': {
    level: 1,
    text: 'This requires a support-ready non-final scenario before the shot. Aim into empty space so the initial supporter pair remains present in both first fixed-step samples. If the verdict says the prerequisite is absent, no different shot can repair that scenario.',
  },
  'support change': {
    level: 2,
    text: 'Load public calibration level 2. Aim at the lowest supported body or its base so the impact removes or changes at least one supporter pair.',
  },
};

function updateReviewGuidance() {
  const goal = document.getElementById('reviewGoal').value;
  const guidance = reviewGuidance[goal];
  document.getElementById('reviewInstructions').textContent = `Recommended level ${guidance.level}. ${guidance.text}`;
  document.getElementById('levelInput').value = guidance.level;
}

function updateReviewSession(session) {
  document.getElementById('reviewState').textContent = session?.state || 'idle';
  const verdict = session?.verdict;
  const verdictEl = document.getElementById('reviewVerdict');
  verdictEl.className = 'review-verdict';
  if (!verdict) {
    verdictEl.textContent = 'No diagnostic capture yet.';
    return;
  }
  verdictEl.classList.add(verdict.status);
  verdictEl.textContent = verdict.demonstrated
    ? `${verdict.goal}: demonstrated${session.eligible_for_issue_44_review ? ' by the frozen confirmatory replay' : ' in the diagnostic pilot only'}.`
    : `${verdict.goal}: ${verdict.status}. ${verdict.reason || ''}`;
}

function updateTelemetry(frame) {
  const state = frame.state?.name || '-';
  if (Number(frame.trajectoryWorldWidth) > 0) trajectoryWorldWidth = Number(frame.trajectoryWorldWidth);
  trajectorySlingCenter = frame.trajectorySlingCenter || null;
  document.getElementById('gameState').textContent = state;
  document.getElementById('currentLevel').textContent = frame.currentLevel ?? '-';
  document.getElementById('numberOfLevels').textContent = frame.numberOfLevels ?? '-';
  document.getElementById('score').textContent = frame.score ?? '-';
}

function drawCrosshair() {
  if (!selectedPoint) return;
  const { canvasX, canvasY } = selectedPoint;
  ctx.save();
  ctx.strokeStyle = '#65d6b5';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(canvasX - 10, canvasY);
  ctx.lineTo(canvasX + 10, canvasY);
  ctx.moveTo(canvasX, canvasY - 10);
  ctx.lineTo(canvasX, canvasY + 10);
  ctx.stroke();
  ctx.restore();
}

function drawAimPreview() {
  if (!aimStartPoint || !aimCurrentPoint) return;
  const slingCenter = previewSlingCenterPoint(aimStartPoint);
  const release = cappedReleasePoint(slingCenter, aimCurrentPoint);
  ctx.save();
  ctx.strokeStyle = '#9bf3d7';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(slingCenter.canvasX, slingCenter.canvasY);
  ctx.lineTo(release.canvasX, release.canvasY);
  ctx.stroke();
  ctx.restore();
}

function previewSlingCenterPoint(fallbackPoint = null) {
  if (trajectorySlingCenter && Number.isFinite(trajectorySlingCenter.canvasX) && Number.isFinite(trajectorySlingCenter.canvasY)) {
    return {
      canvasX: trajectorySlingCenter.canvasX,
      canvasY: trajectorySlingCenter.canvasY,
    };
  }
  return fallbackPoint;
}

function canvasPixelsPerWorldUnit() {
  return canvas.width / trajectoryWorldWidth;
}

function rawCappedReleasePoint(startPoint, releasePoint) {
  const radius = canvasPixelsPerWorldUnit() * WEB_TRAJECTORY_DRAG_RADIUS_WORLD;
  const dx = releasePoint.canvasX - startPoint.canvasX;
  const dy = releasePoint.canvasY - startPoint.canvasY;
  const distance = Math.hypot(dx, dy);
  if (distance <= radius || distance === 0) return releasePoint;
  const scale = radius / distance;
  return {
    canvasX: startPoint.canvasX + dx * scale,
    canvasY: startPoint.canvasY + dy * scale,
  };
}

function cappedReleasePoint(startPoint, releasePoint) {
  const rawRelease = rawCappedReleasePoint(startPoint, releasePoint);
  return {
    canvasX: rawRelease.canvasX,
    canvasY: rawRelease.canvasY + WEB_TRAJECTORY_CANVAS_Y_OFFSET,
    rawCanvasY: rawRelease.canvasY,
  };
}

function buildTrajectoryPreviewPoints(startPoint, releasePoint) {
  const slingCenter = previewSlingCenterPoint(startPoint);
  if (!slingCenter) return [];
  const pixelsPerWorldUnit = canvasPixelsPerWorldUnit();
  const dragRadius = pixelsPerWorldUnit * WEB_TRAJECTORY_DRAG_RADIUS_WORLD;
  const rawRelease = rawCappedReleasePoint(slingCenter, releasePoint);
  const release = cappedReleasePoint(slingCenter, releasePoint);
  const diffX = slingCenter.canvasX - rawRelease.canvasX;
  const diffY = slingCenter.canvasY - rawRelease.canvasY;
  const pullDistance = Math.hypot(diffX, diffY);
  if (pullDistance === 0) return [];

  const velocityMagnitude = (pullDistance / dragRadius) * WEB_TRAJECTORY_MAX_LAUNCH_SPEED * pixelsPerWorldUnit;
  const velocity = {
    x: (diffX / pullDistance) * velocityMagnitude,
    y: (diffY / pullDistance) * velocityMagnitude,
  };
  const gravityY = 9.8 * WEB_TRAJECTORY_LAUNCH_GRAVITY * pixelsPerWorldUnit;
  const timeStep = WEB_TRAJECTORY_TIME_STEP;
  const points = [];
  let position = { ...release };
  let velocityY = velocity.y;

  for (let i = 0; i < WEB_TRAJECTORY_STEPS; i += 1) {
    points.push({ canvasX: position.canvasX, canvasY: position.canvasY });
    position = {
      canvasX: position.canvasX + velocity.x * timeStep,
      canvasY: position.canvasY + velocityY * timeStep + 0.5 * gravityY * timeStep * timeStep,
    };
    velocityY += gravityY * timeStep;
  }
  return points;
}

function drawTrajectoryPreview() {
  if (!aimStartPoint || !aimCurrentPoint) return;
  const points = buildTrajectoryPreviewPoints(aimStartPoint, aimCurrentPoint);
  if (!points.length) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.82)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  let pathStarted = false;
  for (const point of points) {
    if (point.canvasX < 0 || point.canvasX >= canvas.width || point.canvasY < 0 || point.canvasY >= canvas.height) continue;
    if (!pathStarted) {
      ctx.moveTo(point.canvasX, point.canvasY);
      pathStarted = true;
    } else {
      ctx.lineTo(point.canvasX, point.canvasY);
    }
  }
  if (pathStarted) ctx.stroke();
  ctx.restore();
}

function redrawCanvas() {
  if (latestFrameImage) {
    ctx.putImageData(latestFrameImage, 0, 0);
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  drawCrosshair();
  drawAimPreview();
  drawTrajectoryPreview();
}

function clientToCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const rawX = Math.round((event.clientX - rect.left) * (canvas.width / rect.width));
  const rawY = Math.round((event.clientY - rect.top) * (canvas.height / rect.height));
  const canvasX = Math.max(0, Math.min(canvas.width - 1, rawX));
  const canvasY = Math.max(0, Math.min(canvas.height - 1, rawY));
  return { canvasX, canvasY };
}

function canvasPointToGamePoint(point) {
  const canvasY = point.rawCanvasY ?? point.canvasY;
  return { x: point.canvasX, y: canvas.height - 1 - canvasY };
}

function buildAgentActionFromDrag(startPoint, releasePoint) {
  const start = canvasPointToGamePoint(startPoint);
  const release = canvasPointToGamePoint(releasePoint);
  return {
    action_type: 'drag_hold_release',
    coordinate_frame: 'slingshot_relative',
    drag_start: [start.x, start.y],
    drag_release: [release.x - start.x, start.y - release.y],
    holdTime: Number(document.getElementById('holdTime').value),
    tapTime: Number(document.getElementById('tapTime').value),
  };
}

function drawFrame(frame) {
  const binary = atob(frame.rgbBase64);
  const image = ctx.createImageData(frame.width, frame.height);
  for (let i = 0, j = 0; i < binary.length; i += 3, j += 4) {
    image.data[j] = binary.charCodeAt(i);
    image.data[j + 1] = binary.charCodeAt(i + 1);
    image.data[j + 2] = binary.charCodeAt(i + 2);
    image.data[j + 3] = 255;
  }
  canvas.width = frame.width;
  canvas.height = frame.height;
  latestFrameImage = image;
  updateTelemetry(frame);
  redrawCanvas();
  emptyState.classList.add('hidden');
}

async function refreshFrame() {
  const frame = await api('/api/frame');
  drawFrame(frame);
  log('Frame refreshed.');
}

async function refreshStatus() {
  const status = await api('/api/status');
  setStatus(status);
  if (status.preflightErrors?.length) {
    log(`Preflight: ${status.preflightErrors[0]}`);
  }
}

async function run(label, action, afterFrame = false) {
  try {
    const result = await action();
    if (result.connected !== undefined) setStatus(result);
    log(`${label}: ok`);
    if (afterFrame) await refreshFrame();
  } catch (error) {
    log(`${label}: ${error.message}`);
  }
}

function fillShotFields(point) {
  const { canvasX } = point;
  const gamePoint = canvasPointToGamePoint(point);
  selectedPoint = point;
  document.getElementById('shotX').value = gamePoint.x;
  document.getElementById('shotY').value = gamePoint.y;
  log(`Selected shot point x=${canvasX}, y=${gamePoint.y}.`);
}

function scheduleAgentActionTransfer(action) {
  const fast = document.getElementById('fastShot').checked;
  const autoExecute = document.getElementById('autoExecuteAgentAction').checked;
  if (physicsReviewEnabled) {
    pendingReviewAction = { ...action, frame_height: canvas.height };
    document.getElementById('reviewActionSummary').textContent = `Staged drag ${JSON.stringify(action.drag_release)}, hold ${action.holdTime} ms, tap ${action.tapTime} ms. Click “Stage action”; releasing the pointer did not shoot.`;
    log(`Physics-v2 candidate staged locally: ${JSON.stringify(pendingReviewAction)}`);
    return;
  }
  run('Agent action transfer', async () => {
    const result = await post('/api/agent-action', { action, fast });
    log(`Agent action validated: ${JSON.stringify(action)}`);
    if (autoExecute) await post('/api/shot', { ...result.shot, async: true });
    return result;
  });
}

function worldPointsForShape(shape) {
  if (shape.kind === 'circle') {
    const [x, y] = shape.center;
    return [[x - shape.radius, y - shape.radius], [x + shape.radius, y + shape.radius]];
  }
  if (shape.kind === 'box' || shape.kind === 'capsule') {
    const [x, y] = shape.center;
    return [[x - shape.size[0] / 2, y - shape.size[1] / 2], [x + shape.size[0] / 2, y + shape.size[1] / 2]];
  }
  if (shape.kind === 'polygon') return shape.paths.flat();
  if (shape.kind === 'edge') return shape.points;
  return [];
}

function rotatePoint(point, center, degrees) {
  const radians = (degrees || 0) * Math.PI / 180;
  const dx = point[0] - center[0];
  const dy = point[1] - center[1];
  return [center[0] + dx * Math.cos(radians) - dy * Math.sin(radians), center[1] + dx * Math.sin(radians) + dy * Math.cos(radians)];
}

function drawWorldStep(step) {
  const worldCanvas = document.getElementById('worldCanvas');
  const worldCtx = worldCanvas.getContext('2d');
  worldCtx.clearRect(0, 0, worldCanvas.width, worldCanvas.height);
  const points = step.colliders.flatMap((collider) => worldPointsForShape(collider.shape));
  if (!points.length) return;
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs) - 1;
  const maxX = Math.max(...xs) + 1;
  const minY = Math.min(...ys) - 1;
  const maxY = Math.max(...ys) + 1;
  const scale = Math.min((worldCanvas.width - 40) / Math.max(1, maxX - minX), (worldCanvas.height - 40) / Math.max(1, maxY - minY));
  const toCanvas = ([x, y]) => [20 + (x - minX) * scale, worldCanvas.height - 20 - (y - minY) * scale];

  worldCtx.lineWidth = 2;
  for (const collider of step.colliders) {
    const shape = collider.shape;
    worldCtx.strokeStyle = collider.enabled ? '#65d6b5' : '#52636e';
    if (collider.is_trigger) worldCtx.setLineDash([5, 5]);
    else worldCtx.setLineDash([]);
    worldCtx.beginPath();
    if (shape.kind === 'circle') {
      const center = toCanvas(shape.center);
      worldCtx.arc(center[0], center[1], shape.radius * scale, 0, Math.PI * 2);
    } else if (shape.kind === 'box') {
      const [cx, cy] = shape.center;
      const halfX = shape.size[0] / 2;
      const halfY = shape.size[1] / 2;
      const corners = [[cx - halfX, cy - halfY], [cx + halfX, cy - halfY], [cx + halfX, cy + halfY], [cx - halfX, cy + halfY]]
        .map((point) => toCanvas(rotatePoint(point, shape.center, shape.angle_degrees)));
      corners.forEach((point, index) => index ? worldCtx.lineTo(...point) : worldCtx.moveTo(...point));
      worldCtx.closePath();
    } else if (shape.kind === 'polygon') {
      for (const path of shape.paths) {
        path.map(toCanvas).forEach((point, index) => index ? worldCtx.lineTo(...point) : worldCtx.moveTo(...point));
        worldCtx.closePath();
      }
    } else if (shape.kind === 'edge') {
      shape.points.map(toCanvas).forEach((point, index) => index ? worldCtx.lineTo(...point) : worldCtx.moveTo(...point));
    } else if (shape.kind === 'capsule') {
      const center = toCanvas(shape.center);
      worldCtx.ellipse(center[0], center[1], shape.size[0] * scale / 2, shape.size[1] * scale / 2, -(shape.angle_degrees || 0) * Math.PI / 180, 0, Math.PI * 2);
    }
    worldCtx.stroke();
  }
  worldCtx.setLineDash([]);
  for (const contact of step.contacts) {
    const point = toCanvas(contact.point);
    worldCtx.fillStyle = '#ff8c8c';
    worldCtx.beginPath();
    worldCtx.arc(point[0], point[1], 4, 0, Math.PI * 2);
    worldCtx.fill();
  }
  const bodyPositions = new Map(step.entities.filter((entity) => entity.body_present).map((entity) => [entity.entity_id, entity.body.position]));
  worldCtx.strokeStyle = '#ffd166';
  for (const support of step.supports) {
    const from = bodyPositions.get(support.supporter_entity_id);
    const to = bodyPositions.get(support.supported_entity_id);
    if (!from || !to) continue;
    worldCtx.beginPath();
    worldCtx.moveTo(...toCanvas(from));
    worldCtx.lineTo(...toCanvas(to));
    worldCtx.stroke();
  }
  document.getElementById('reviewStepLabel').textContent = `fixed step ${step.fixed_step}`;
  document.getElementById('reviewStepFacts').textContent = JSON.stringify({
    gravity: step.world.gravity_vector,
    contacts: step.contacts,
    supports: step.supports,
    entities: step.entities.map((entity) => ({
      entity_id: entity.entity_id,
      lifecycle: entity.lifecycle,
      body: entity.body,
    })),
  }, null, 2);
}

async function loadReviewSteps() {
  reviewSteps = [];
  let start = 0;
  let total = 1;
  while (start < total) {
    const result = await api(`/api/physics-v2-review/steps?start=${start}&count=100`);
    reviewSteps.push(...(result.steps || []));
    total = result.total || 0;
    if (!result.count) break;
    start += result.count;
  }
  if (!reviewSteps.length) return;
  document.getElementById('worldPlayback').classList.remove('hidden');
  document.querySelector('.review-workspace').classList.add('with-playback');
  const timeline = document.getElementById('reviewTimeline');
  timeline.min = 0;
  timeline.max = reviewSteps.length - 1;
  timeline.value = 0;
  drawWorldStep(reviewSteps[0]);
}

async function reviewTransition(label, path, afterSteps = false) {
  await run(label, async () => {
    const result = await post(path);
    updateReviewSession(result.session);
    if (afterSteps) await loadReviewSteps();
    return result;
  }, false);
}

canvas.addEventListener('pointerdown', (event) => {
  aimStartPoint = clientToCanvasPoint(event);
  aimCurrentPoint = aimStartPoint;
  canvas.classList.add('aiming');
  canvas.setPointerCapture(event.pointerId);
  redrawCanvas();
});

canvas.addEventListener('pointermove', (event) => {
  if (!aimStartPoint) return;
  aimCurrentPoint = clientToCanvasPoint(event);
  redrawCanvas();
});

canvas.addEventListener('pointerup', (event) => {
  if (!aimStartPoint) return;
  aimCurrentPoint = clientToCanvasPoint(event);
  const slingCenter = previewSlingCenterPoint(aimStartPoint);
  const releasePoint = cappedReleasePoint(slingCenter, aimCurrentPoint);
  const action = buildAgentActionFromDrag(slingCenter, releasePoint);
  fillShotFields(releasePoint);
  scheduleAgentActionTransfer(action);
  aimStartPoint = null;
  aimCurrentPoint = null;
  canvas.classList.remove('aiming');
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  redrawCanvas();
});

canvas.addEventListener('pointercancel', (event) => {
  aimStartPoint = null;
  aimCurrentPoint = null;
  canvas.classList.remove('aiming');
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  redrawCanvas();
});

document.getElementById('startBtn').addEventListener('click', () => run('Start game', () => post('/api/start'), true));
document.getElementById('connectBtn').addEventListener('click', () => run('Connect', () => post('/api/connect'), true));
document.getElementById('stopBtn').addEventListener('click', () => run('Stop', () => post('/api/stop')));
document.getElementById('refreshBtn').addEventListener('click', () => run('Refresh frame', refreshFrame));
document.getElementById('restartBtn').addEventListener('click', () => run('Restart', () => post('/api/restart'), true));
document.getElementById('zoomOutBtn').addEventListener('click', () => run('Zoom out', () => post('/api/zoom-out'), true));
document.getElementById('zoomInBtn').addEventListener('click', () => run('Zoom in', () => post('/api/zoom-in'), true));
document.getElementById('loadLevelBtn').addEventListener('click', () => {
  const level = Number(document.getElementById('levelInput').value);
  run(`Load next available level from ${level}`, () => post('/api/load-level', { level }), true);
});
document.getElementById('shootBtn').addEventListener('click', () => {
  const payload = {
    x: Number(document.getElementById('shotX').value),
    y: Number(document.getElementById('shotY').value),
    tapTime: Number(document.getElementById('tapTime').value),
    releaseTime: Number(document.getElementById('holdTime').value),
    fast: document.getElementById('fastShot').checked,
  };
  run(`Shoot x=${payload.x}, y=${payload.y}`, () => post('/api/shot', payload), true);
});
document.getElementById('autoRefresh').addEventListener('change', (event) => {
  if (event.target.checked) {
    autoTimer = setInterval(() => refreshFrame().catch((error) => log(`Auto-refresh: ${error.message}`)), 1000);
    log('Auto-refresh enabled.');
  } else {
    clearInterval(autoTimer);
    autoTimer = null;
    log('Auto-refresh disabled.');
  }
});

document.getElementById('reviewGoal').addEventListener('change', updateReviewGuidance);
document.getElementById('loadReviewGoalBtn').addEventListener('click', () => run('Load recommended review scenario', async () => {
  const goal = document.getElementById('reviewGoal').value;
  const result = await post('/api/physics-v2-review/load-goal', { goal });
  await refreshFrame();
  return result;
}));
document.getElementById('stageReviewBtn').addEventListener('click', () => run('Stage physics-v2 action', async () => {
  if (!pendingReviewAction) throw new Error('Drag on the game frame first; review mode never shoots on pointer release.');
  const result = await post('/api/physics-v2-review/stage', {
    goal: document.getElementById('reviewGoal').value,
    action: pendingReviewAction,
  });
  updateReviewSession(result.session);
  return result;
}));
document.getElementById('exploreReviewBtn').addEventListener('click', () => reviewTransition('Exploratory physics-v2 shot', '/api/physics-v2-review/explore', true));
document.getElementById('freezeReviewBtn').addEventListener('click', () => reviewTransition('Freeze exact replay', '/api/physics-v2-review/freeze'));
document.getElementById('replayReviewBtn').addEventListener('click', () => reviewTransition('Confirmatory physics-v2 replay', '/api/physics-v2-review/replay', true));
document.getElementById('reviewTimeline').addEventListener('input', (event) => {
  const step = reviewSteps[Number(event.target.value)];
  if (step) drawWorldStep(step);
});

updateReviewGuidance();

refreshStatus().catch((error) => log(`Status: ${error.message}`));
