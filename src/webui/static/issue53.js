const state = {
  review: null,
  selected: null,
  detail: null,
  steps: [],
  stepIndex: 0,
  timer: null,
  bounds: null,
};

const byId = (id) => document.getElementById(id);

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function toast(message, failed = false) {
  const element = byId('toast');
  element.textContent = message;
  element.className = failed ? 'visible failed' : 'visible';
  window.setTimeout(() => { element.className = ''; }, 3500);
}

async function loadReview() {
  const payload = await request('/api/issue-53-review');
  state.review = payload.review;
  renderReview();
}

function renderReview() {
  const review = state.review;
  byId('progress').textContent = `${review.reviewProgress.decided}/${review.mismatchCount} decided · ${review.reviewProgress.opened}/${review.mismatchCount} opened`;
  byId('accessState').textContent = review.authorizedFinalAccess ? 'Final access audited' : 'Final locked';
  byId('accessState').className = review.authorizedFinalAccess ? 'pill authorized' : 'pill locked';
  byId('authorizationPanel').classList.toggle('hidden', review.authorizedFinalAccess);
  const playlist = byId('playlist');
  playlist.replaceChildren();
  for (const item of review.playlist) {
    const button = document.createElement('button');
    button.className = `playlist-item${state.selected === item.index ? ' selected' : ''}${item.locked ? ' locked-item' : ''}`;
    button.disabled = item.locked;
    if (item.locked) {
      button.innerHTML = `<span class="item-number">${item.index + 1}</span><span><strong>Sealed final item</strong><small>Authorization required</small></span><span class="item-status">Locked</span>`;
    } else {
      const status = item.decision ? item.decision.decision.replaceAll('_', ' ') : (item.traceOpened ? 'Opened' : 'Unopened');
      const action = item.socketCommand;
      button.innerHTML = `<span class="item-number">${item.index + 1}</span><span><strong>${item.role.replaceAll('_', ' ')} · ${item.scenarioFamily}</strong><small>${item.intendedStratum} · shot (${action.x}, ${action.y})</small><small>${item.expectedTermination} → ${item.observedTermination}</small></span><span class="item-status">${status}</span>`;
      button.addEventListener('click', () => selectItem(item.index));
    }
    playlist.appendChild(button);
  }
  if (review.followUpDecision) toast(`${review.followUpDecision.outcome}: ${review.followUpDecision.nextStep}`);
}

async function authorizeFinal() {
  const identity = byId('authorizationIdentity').value.trim();
  const payload = await request('/api/issue-53-review/authorize', {
    method: 'POST', body: JSON.stringify({ authorizationIdentity: identity }),
  });
  state.review = payload.review;
  renderReview();
  toast('Final review access validated and recorded under sealed/access.');
}

async function selectItem(index) {
  stopPlayback();
  state.selected = index;
  state.steps = [];
  state.bounds = null;
  const payload = await request(`/api/issue-53-review/items/${index}/open`, { method: 'POST', body: '{}' });
  state.detail = payload.detail;
  await loadAllSteps(index, payload.detail.fixedStepCount);
  state.stepIndex = 0;
  renderReview();
  renderDetail();
}

async function loadAllSteps(index, total) {
  const pageSize = 120;
  for (let start = 0; start < total; start += pageSize) {
    const page = await request(`/api/issue-53-review/items/${index}/steps?start=${start}&count=${pageSize}`);
    state.steps.push(...page.steps);
  }
  state.bounds = playbackBounds(state.steps);
}

function renderDetail() {
  const detail = state.detail;
  const item = detail.item;
  byId('emptyReview').classList.add('hidden');
  byId('reviewContent').classList.remove('hidden');
  byId('itemRole').textContent = item.role.replaceAll('_', ' ');
  byId('itemTitle').textContent = `${item.intendedStratum} termination mismatch`;
  byId('expectedTermination').textContent = item.expectedTermination;
  byId('observedTermination').textContent = item.observedTermination;
  byId('scenarioFamily').textContent = item.scenarioFamily;
  byId('intendedStratum').textContent = item.intendedStratum;
  const command = item.socketCommand;
  byId('socketCommand').textContent = `x=${command.x}, y=${command.y}, hold=${command.releaseTime}ms, tap=${command.tapTime}ms`;
  byId('coverageFacts').textContent = item.coverageFacts.join(', ');
  byId('terminationExplanation').textContent = detail.terminationExplanation;
  const timeline = byId('timeline');
  timeline.max = Math.max(0, state.steps.length - 1);
  timeline.value = 0;
  renderStep();
  renderReplay(item);
  const decision = item.decision;
  byId('decisionStatus').textContent = decision ? `Recorded: ${decision.decision.replaceAll('_', ' ')} — ${decision.notes || 'No notes.'}` : 'No decision recorded.';
  for (const id of ['reviewer', 'decision', 'notes', 'recordDecisionBtn']) byId(id).disabled = Boolean(decision);
}

function playbackBounds(steps) {
  const points = [];
  for (const step of steps) {
    for (const entity of step.entities) if (entity.position) points.push(entity.position);
    for (const contact of step.contacts) points.push(contact.point);
  }
  if (!points.length) return { minX: -15, maxX: 15, minY: -6, maxY: 6 };
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  return { minX: Math.min(...xs) - 1, maxX: Math.max(...xs) + 1, minY: Math.min(...ys) - 1, maxY: Math.max(...ys) + 1 };
}

function worldPoint(position, canvas) {
  const bounds = state.bounds;
  const scale = Math.min((canvas.width - 40) / (bounds.maxX - bounds.minX), (canvas.height - 40) / (bounds.maxY - bounds.minY));
  return {
    x: 20 + (position[0] - bounds.minX) * scale,
    y: canvas.height - 20 - (position[1] - bounds.minY) * scale,
  };
}

function entityColor(kind, lifecycle) {
  if (lifecycle !== 'active') return '#66717d';
  return ({ pig: '#72d48c', bird: '#f26b5e', block: '#dca95d', platform: '#8aa4ff', slingshot: '#a975e6' })[kind] || '#b7c2cc';
}

function renderStep() {
  const step = state.steps[state.stepIndex];
  if (!step) return;
  byId('timeline').value = state.stepIndex;
  byId('stepLabel').textContent = `Fixed step ${step.fixedStep} · ${state.stepIndex + 1}/${state.steps.length}`;
  byId('pigsRemaining').textContent = step.pigsRemaining;
  byId('birdsRemaining').textContent = step.birdsRemaining;
  byId('contactCount').textContent = step.contacts.length;
  byId('supportCount').textContent = step.supports.length;
  byId('eventOverlay').textContent = step.events.length ? step.events.map((event) => `${event.event_type}: ${event.participants.join(' ↔ ') || 'level'}`).join(' · ') : 'No event at this fixed step.';

  const canvas = byId('traceCanvas');
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#111821';
  context.fillRect(0, 0, canvas.width, canvas.height);
  const positions = new Map(step.entities.filter((entity) => entity.position).map((entity) => [entity.id, worldPoint(entity.position, canvas)]));
  context.lineWidth = 2;
  for (const support of step.supports) {
    const from = positions.get(support.supporter);
    const to = positions.get(support.supported);
    if (!from || !to) continue;
    context.strokeStyle = '#72d48c'; context.beginPath(); context.moveTo(from.x, from.y); context.lineTo(to.x, to.y); context.stroke();
  }
  for (const contact of step.contacts) {
    const point = worldPoint(contact.point, canvas);
    context.fillStyle = '#ffd166'; context.beginPath(); context.arc(point.x, point.y, 3.2, 0, Math.PI * 2); context.fill();
  }
  for (const entity of step.entities) {
    if (!entity.position) continue;
    const point = positions.get(entity.id);
    context.save(); context.translate(point.x, point.y); context.rotate(-(entity.rotation || 0) * Math.PI / 180);
    context.fillStyle = entityColor(entity.kind, entity.lifecycle);
    if (entity.kind === 'block' || entity.kind === 'platform') context.fillRect(-9, -6, 18, 12);
    else { context.beginPath(); context.arc(0, 0, entity.kind === 'pig' ? 8 : 6, 0, Math.PI * 2); context.fill(); }
    context.restore();
    context.fillStyle = '#dce6ef'; context.font = '10px system-ui'; context.fillText(entity.id.replace('runtime:', ''), point.x + 9, point.y - 8);
  }
}

function togglePlayback() {
  if (state.timer) { stopPlayback(); return; }
  byId('playBtn').textContent = 'Pause';
  state.timer = window.setInterval(() => {
    if (state.stepIndex >= state.steps.length - 1) { stopPlayback(); return; }
    state.stepIndex += 1; renderStep();
  }, 20);
}

function stopPlayback() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  if (byId('playBtn')) byId('playBtn').textContent = 'Play';
}

function renderReplay(item) {
  const replay = item.replay;
  const contract = state.review.screenCoordinateAlignmentContract;
  const alignment = replay && replay.screen_coordinate_alignment;
  byId('alignmentEvidence').textContent = alignment
    ? `Alignment proven: startup ${alignment.startup_speed}× → execution ${alignment.execution_speed}×; retained anchor matched within ${alignment.alignment_contract.anchor_tolerance_pixels}px.`
    : (contract
      ? `Required before shot: ${contract.stable_observations_required_per_phase} stable samples at ${contract.startup_speed}× and execution speed; retained anchor within ${contract.retained_anchor_tolerance_pixels}px; frozen command unchanged.`
      : 'Screen-coordinate alignment contract unavailable; replay is blocked.');
  const video = byId('replayVideo');
  const empty = byId('videoEmpty');
  if (!replay) {
    video.removeAttribute('src'); video.classList.add('hidden'); empty.classList.remove('hidden');
    byId('replayState').textContent = 'Not run';
    byId('replayBtn').disabled = Boolean(item.decision) || !contract;
    byId('comparisonFacts').textContent = 'No original/replay comparison yet.';
    return;
  }
  video.src = `/api/issue-53-review/items/${item.index}/video`; video.classList.remove('hidden'); empty.classList.add('hidden');
  byId('replayState').textContent = replay.comparison && replay.comparison.passed ? 'Agrees' : 'Investigation required';
  byId('replayBtn').disabled = true;
  const components = replay.comparison ? replay.comparison.components : [];
  byId('comparisonFacts').textContent = components.map((component) => `${component.component}: ${component.status}`).join(' · ');
}

async function runReplay() {
  byId('replayBtn').disabled = true;
  byId('replayState').textContent = 'Running…';
  const payload = await request(`/api/issue-53-review/items/${state.selected}/replay`, { method: 'POST', body: '{}' });
  state.detail = payload.detail;
  await loadReview();
  renderDetail();
  toast('Diagnostic replay retained separately and excluded from production accounting.');
}

async function recordDecision() {
  const payload = await request(`/api/issue-53-review/items/${state.selected}/decision`, {
    method: 'POST',
    body: JSON.stringify({ reviewer: byId('reviewer').value, decision: byId('decision').value, notes: byId('notes').value }),
  });
  state.review = payload.review;
  const selected = state.review.playlist.find((item) => item.index === state.selected);
  state.detail.item = selected;
  renderReview(); renderDetail();
  toast('Immutable decision recorded outside production evidence.');
}

byId('authorizeBtn').addEventListener('click', () => authorizeFinal().catch((error) => toast(error.message, true)));
byId('playBtn').addEventListener('click', togglePlayback);
byId('previousBtn').addEventListener('click', () => { state.stepIndex = Math.max(0, state.stepIndex - 1); renderStep(); });
byId('nextBtn').addEventListener('click', () => { state.stepIndex = Math.min(state.steps.length - 1, state.stepIndex + 1); renderStep(); });
byId('timeline').addEventListener('input', (event) => { state.stepIndex = Number(event.target.value); renderStep(); });
byId('replayBtn').addEventListener('click', () => runReplay().catch((error) => { byId('replayBtn').disabled = false; toast(error.message, true); }));
byId('recordDecisionBtn').addEventListener('click', () => recordDecision().catch((error) => toast(error.message, true)));
loadReview().catch((error) => toast(error.message, true));
