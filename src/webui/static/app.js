const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const emptyState = document.getElementById('emptyState');
const logEl = document.getElementById('log');
let selectedPoint = null;
let aimStartPoint = null;
let aimCurrentPoint = null;
let latestFrameImage = null;
let autoTimer = null;

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
}

function updateTelemetry(frame) {
  const state = frame.state?.name || '-';
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
  ctx.save();
  ctx.strokeStyle = '#9bf3d7';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(aimStartPoint.canvasX, aimStartPoint.canvasY);
  ctx.lineTo(aimCurrentPoint.canvasX, aimCurrentPoint.canvasY);
  ctx.stroke();
  ctx.restore();
}

function drawTrajectoryPreview() {
  if (!aimStartPoint || !aimCurrentPoint) return;
  const velocityX = aimStartPoint.canvasX - aimCurrentPoint.canvasX;
  const velocityY = aimStartPoint.canvasY - aimCurrentPoint.canvasY;
  ctx.save();
  ctx.fillStyle = 'rgba(155, 243, 215, 0.78)';
  for (let i = 1; i <= 12; i += 1) {
    const time = i / 3;
    const x = aimStartPoint.canvasX + velocityX * time * 0.18;
    const y = aimStartPoint.canvasY + velocityY * time * 0.18 + time * time * 5;
    if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) continue;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
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
  return { x: point.canvasX, y: canvas.height - 1 - point.canvasY };
}

function buildAgentActionFromDrag(startPoint, releasePoint) {
  const start = canvasPointToGamePoint(startPoint);
  const release = canvasPointToGamePoint(releasePoint);
  return {
    action_type: 'drag_release',
    coordinate_frame: 'slingshot_relative',
    drag_start: [start.x, start.y],
    drag_release: [release.x - start.x, start.y - release.y],
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
  redrawCanvas();
  emptyState.classList.add('hidden');
  updateTelemetry(frame);
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
  const { canvasX, canvasY } = point;
  const gamePoint = canvasPointToGamePoint(point);
  const gameY = canvas.height - 1 - canvasY;
  selectedPoint = point;
  document.getElementById('shotX').value = gamePoint.x;
  document.getElementById('shotY').value = gameY;
  log(`Selected shot point x=${canvasX}, y=${gameY}.`);
}

function scheduleAgentActionTransfer(action) {
  const fast = document.getElementById('fastShot').checked;
  const autoExecute = document.getElementById('autoExecuteAgentAction').checked;
  run('Agent action transfer', async () => {
    const result = await post('/api/agent-action', { action, fast });
    log(`Agent action validated: ${JSON.stringify(action)}`);
    if (autoExecute) await post('/api/shot', { ...result.shot, async: true });
    return result;
  });
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
  const action = buildAgentActionFromDrag(aimStartPoint, aimCurrentPoint);
  fillShotFields(aimCurrentPoint);
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

refreshStatus().catch((error) => log(`Status: ${error.message}`));
