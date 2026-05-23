/**
 * GestureSense
 * Real-time hand gesture and facial-expression demo.
 *
 * Created and maintained by Saad Kamal.
 *
 * This file owns the runtime pipeline:
 * camera -> canvas frame -> MediaPipe models -> gesture/expression classifiers
 * -> overlay drawing and HUD updates.
 */
import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
import {
  EXPRESSIONS,
  FACE_OVAL,
  classifyExpression,
  stabilizeExpression,
} from './expressions.js?v=37';

const startScreen = document.getElementById('start-screen');
const trackingScreen = document.getElementById('tracking-screen');
const startBtn = document.getElementById('start-btn');
const video = document.getElementById('webcam');
const viewCanvas = document.getElementById('view');
const viewCtx = viewCanvas.getContext('2d');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const loading = document.getElementById('loading');
const fpsEl = document.getElementById('fps');
const handCountEl = document.getElementById('hand-count');
const detectCountEl = document.getElementById('detect-count');
const gestureLabel = document.getElementById('gesture-label');
const handPanels = document.getElementById('hand-panels');
const streamStatusEl = document.getElementById('stream-status');
const expressionPanel = document.getElementById('expression-panel');
const expressionEmoji = document.getElementById('expression-emoji');
const expressionLabel = document.getElementById('expression-label');
const expressionConfidence = document.getElementById('expression-confidence');
const handPanelSlots = {
  A: document.querySelector('[data-hand-slot="A"]'),
  B: document.querySelector('[data-hand-slot="B"]'),
};

const FINGERS = {
  thumb:  { tip: 4, ip: 3, mcp: 2, name: 'Thumb'  },
  index:  { tip: 8, pip: 6, mcp: 5, name: 'Index'  },
  middle: { tip: 12, pip: 10, mcp: 9, name: 'Middle' },
  ring:   { tip: 16, pip: 14, mcp: 13, name: 'Ring'   },
  pinky:  { tip: 20, pip: 18, mcp: 17, name: 'Pinky'  },
};

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

// Hand A/B colors are intentionally stable. They represent tracker identity,
// not MediaPipe's per-frame handedness, which can flicker when hands overlap.
const HAND_COLORS = {
  Left:  { primary: '#00f0ff', secondary: '#8338ec', glow: 'rgba(0,240,255,0.35)' },
  Right: { primary: '#ff006e', secondary: '#8338ec', glow: 'rgba(255,0,110,0.35)' },
};

const SMOOTH_ALPHA = 0.45;
const SMOOTH_ALPHA_CLOSE = 0.22;
const CLOSE_HANDS_DIST = 0.14;
const TIP_INDICES = [4, 8, 12, 16, 20];
const HAND_HISTORY_FRAMES = 24;
const isEmbeddedBrowser = /Electron/i.test(navigator.userAgent);

const TRACKER_COLORS = [
  { primary: '#00f0ff', secondary: '#8338ec', glow: 'rgba(0,240,255,0.35)', label: 'A' },
  { primary: '#ff006e', secondary: '#8338ec', glow: 'rgba(255,0,110,0.35)', label: 'B' },
];

const detectCanvas = document.createElement('canvas');
const detectCtx = detectCanvas.getContext('2d', { willReadFrequently: true });

let handLandmarker = null;
let faceLandmarker = null;
let running = false;
let mediaStream = null;
let currentHands = [];
let handTrackers = [];
let nextTrackerId = 0;
let lastPairAssignment = [0, 1];
let particles = [];
let detectIntervalId = null;
let fpsLastTick = performance.now();
let frameCount = 0;
let detectFrameCount = 0;
let fps = 0;
let detectFps = 0;
let lastFrameSig = -1;
let staleFrameCount = 0;
let hudCache = { handCount: -1, gesture: '', panelsKey: '', expression: '' };
let faceState = null;
let stableFaceExpression = null;

function dist2(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function dist3(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function jointAngle(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const v2 = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 * m2 === 0) return 0;
  const cos = Math.max(-1, Math.min(1, (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / (m1 * m2)));
  return Math.acos(cos);
}

function isThumbExtended(landmarks) {
  const tip = landmarks[4];
  const ip = landmarks[3];
  const mcp = landmarks[2];
  const wrist = landmarks[0];
  const indexMcp = landmarks[5];
  const indexPip = landmarks[6];
  const middleMcp = landmarks[9];
  const pinkyMcp = landmarks[17];

  const palmWidth = dist2(indexMcp, pinkyMcp);
  const tipToIndexMcp = dist2(tip, indexMcp);
  const tipToMiddleMcp = dist2(tip, middleMcp);
  const thumbAngle = jointAngle(mcp, ip, tip);
  const tipFromWrist = dist3(tip, wrist);
  const ipFromWrist = dist3(ip, wrist);

  const palm = {
    x: (indexMcp.x + pinkyMcp.x) / 2,
    y: (indexMcp.y + pinkyMcp.y) / 2,
    z: (indexMcp.z + pinkyMcp.z) / 2,
  };
  const tipFromPalm = dist3(tip, palm);
  const ipFromPalm = dist3(ip, palm);
  const mcpFromPalm = dist3(mcp, palm);

  // ── Tucked / hidden in fist — reject early ──
  if (tipToIndexMcp < palmWidth * 0.62) return false;
  if (tipToMiddleMcp < palmWidth * 0.55) return false;
  if (thumbAngle < 2.35) return false;
  if (tipFromWrist < ipFromWrist * 1.08) return false;
  if (tipFromPalm < mcpFromPalm * 1.05) return false;

  // Thumb tip sitting over curled index (common tucked pose)
  if (tip.y > indexPip.y && tipToIndexMcp < palmWidth * 0.85) return false;

  // ── Must show clear extension (need 3 of 5) ──
  let score = 0;
  if (tipFromWrist > ipFromWrist * 1.12) score++;
  if (thumbAngle > 2.5) score++;
  if (tip.z < ip.z - 0.018) score++;
  if (tipFromPalm > ipFromPalm * 1.1) score++;
  if (tipToIndexMcp > palmWidth * 0.78) score++;

  return score >= 3;
}

function isFingerExtended(landmarks, finger) {
  if (finger === 'thumb') return isThumbExtended(landmarks);
  const { tip, pip, mcp } = FINGERS[finger];
  return landmarks[tip].y < landmarks[pip].y - 0.012 && landmarks[pip].y < landmarks[mcp].y;
}

function getFingerStates(landmarks, tracker) {
  const raw = {};
  for (const key of Object.keys(FINGERS)) raw[key] = isFingerExtended(landmarks, key);

  if (!tracker.fingerStates) {
    tracker.fingerStates = { ...raw };
    return tracker.fingerStates;
  }

  const prev = tracker.fingerStates;
  for (const key of Object.keys(FINGERS)) {
    if (raw[key] === prev[key]) {
      prev[`_${key}Streak`] = 0;
      continue;
    }
    prev[`_${key}Streak`] = (prev[`_${key}Streak`] || 0) + 1;
    const needed = key === 'thumb' ? (raw[key] ? 3 : 2) : 2;
    if (prev[`_${key}Streak`] >= needed) {
      prev[key] = raw[key];
      prev[`_${key}Streak`] = 0;
    }
  }
  return prev;
}

function landmarkCloudDistance(a, b) {
  if (!a || !b) return Infinity;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += dist3(a[i], b[i]);
  return sum / a.length;
}

function matchCost(detection, tracker) {
  const wristDist = dist2(detection.raw[0], tracker.wrist);
  const cloudDist = landmarkCloudDistance(detection.raw, tracker.landmarks);
  const handPenalty = detection.handedness !== tracker.stableHandedness ? 0.04 : 0;
  return wristDist * 2.5 + cloudDist + handPenalty;
}

// Smooth landmarks on the persistent tracker rather than on raw frame order.
// MediaPipe may return hands in a different order when they are close together.
function smoothIntoTracker(raw, tracker, alpha) {
  if (!tracker.landmarks) {
    tracker.landmarks = raw.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }));
    tracker.wrist = { x: raw[0].x, y: raw[0].y };
    return tracker.landmarks;
  }

  for (let i = 0; i < raw.length; i++) {
    tracker.landmarks[i].x += alpha * (raw[i].x - tracker.landmarks[i].x);
    tracker.landmarks[i].y += alpha * (raw[i].y - tracker.landmarks[i].y);
    tracker.landmarks[i].z += alpha * (raw[i].z - tracker.landmarks[i].z);
  }
  tracker.wrist.x += alpha * (raw[0].x - tracker.wrist.x);
  tracker.wrist.y += alpha * (raw[0].y - tracker.wrist.y);
  return tracker.landmarks;
}

function updateTrackerHandedness(tracker, handedness) {
  if (handedness === tracker.stableHandedness) {
    tracker.handedness = handedness;
    tracker.handednessStreak = 0;
    return;
  }
  tracker.handednessStreak = (tracker.handednessStreak || 0) + 1;
  if (tracker.handednessStreak >= 5) {
    tracker.stableHandedness = handedness;
    tracker.handedness = handedness;
    tracker.handednessStreak = 0;
  }
}

function createTracker(detection) {
  const id = nextTrackerId++;
  const tracker = {
    id,
    color: TRACKER_COLORS[id % TRACKER_COLORS.length],
    landmarks: null,
    wrist: { x: detection.raw[0].x, y: detection.raw[0].y },
    handedness: detection.handedness,
    stableHandedness: detection.handedness,
    handednessStreak: 0,
    fingerStates: null,
    history: [],
    signal: null,
    missed: 0,
  };
  smoothIntoTracker(detection.raw, tracker, 1);
  getFingerStates(tracker.landmarks, tracker);
  updateHandSignal(tracker);
  handTrackers.push(tracker);
  return tracker;
}

function assignTwoHands(detections, alpha) {
  const [d0, d1] = detections;
  const [t0, t1] = handTrackers;

  const same = matchCost(d0, t0) + matchCost(d1, t1);
  const cross = matchCost(d0, t1) + matchCost(d1, t0);

  const prevSame = lastPairAssignment[0] === 0 && lastPairAssignment[1] === 1;
  const currentCost = prevSame ? same : cross;
  const bestCost = Math.min(same, cross);
  let useSame;

  if (bestCost < currentCost * 0.82) {
    useSame = same <= cross;
  } else {
    useSame = prevSame;
  }

  lastPairAssignment = useSame ? [0, 1] : [1, 0];
  const pairs = useSame ? [[d0, t0], [d1, t1]] : [[d0, t1], [d1, t0]];

  for (const [det, tracker] of pairs) {
    smoothIntoTracker(det.raw, tracker, alpha);
    updateTrackerHandedness(tracker, det.handedness);
    getFingerStates(tracker.landmarks, tracker);
    updateHandSignal(tracker);
    tracker.missed = 0;
  }
}

function assignDetections(detections, alpha) {
  handTrackers = handTrackers.filter(t => t.missed <= 6);
  if (detections.length >= 2 && handTrackers.length > 2) {
    handTrackers.sort((a, b) => a.missed - b.missed);
    handTrackers.length = 2;
  }

  if (detections.length === 0) return;

  if (handTrackers.length === 0) {
    detections.forEach(d => createTracker(d));
    return;
  }

  if (detections.length === 1) {
    let best = handTrackers[0];
    let bestCost = matchCost(detections[0], best);
    for (let i = 1; i < handTrackers.length; i++) {
      const c = matchCost(detections[0], handTrackers[i]);
      if (c < bestCost) {
        bestCost = c;
        best = handTrackers[i];
      }
    }
    smoothIntoTracker(detections[0].raw, best, alpha);
    updateTrackerHandedness(best, detections[0].handedness);
    getFingerStates(best.landmarks, best);
    updateHandSignal(best);
    best.missed = 0;
    for (const t of handTrackers) {
      if (t !== best) t.missed++;
    }
    return;
  }

  if (detections.length >= 2 && handTrackers.length === 1) {
    const t0 = handTrackers[0];
    const c0 = matchCost(detections[0], t0);
    const c1 = matchCost(detections[1], t0);
    const [near, far] = c0 <= c1 ? [detections[0], detections[1]] : [detections[1], detections[0]];
    smoothIntoTracker(near.raw, t0, alpha);
    updateTrackerHandedness(t0, near.handedness);
    getFingerStates(t0.landmarks, t0);
    updateHandSignal(t0);
    t0.missed = 0;
    createTracker(far);
    lastPairAssignment = [0, 1];
    return;
  }

  if (detections.length >= 2 && handTrackers.length >= 2) {
    while (handTrackers.length < 2) createTracker(detections[handTrackers.length]);
    assignTwoHands(detections.slice(0, 2), alpha);
    for (let i = 2; i < detections.length; i++) createTracker(detections[i]);
  }
}

function buildCurrentHands() {
  currentHands = handTrackers
    .filter(t => t.missed <= 2 && t.landmarks)
    .sort((a, b) => a.id - b.id)
    .map(t => ({
      landmarks: t.landmarks,
      handedness: t.handedness,
      stableHandedness: t.stableHandedness,
      fingerStates: t.fingerStates,
      signal: t.signal,
      color: t.color,
      id: t.id,
    }));
}

// Static and motion gestures are intentionally heuristic. MediaPipe gives us
// landmarks; GestureSense turns them into demo-friendly semantic signals.
function isHandRaised(landmarks) {
  return landmarks[12].y < landmarks[0].y - 0.05;
}

function describeGesture(fingerStates) {
  const up = Object.keys(FINGERS).filter(k => fingerStates[k]);
  if (up.length === 0) return 'Hand detected — fingers closed';
  if (up.length === 5) return 'Open palm — all fingers extended';
  if (up.length === 1 && up[0] === 'index') return 'Pointing gesture detected';
  if (up.length === 2 && up.includes('index') && up.includes('middle')) return 'Peace sign detected';
  if (up.length === 1 && up[0] === 'thumb') return 'Thumbs up detected';
  return `${up.length} finger${up.length > 1 ? 's' : ''} raised`;
}

function handCenter(landmarks) {
  const ids = [0, 5, 9, 13, 17];
  return ids.reduce((acc, i) => {
    acc.x += landmarks[i].x / ids.length;
    acc.y += landmarks[i].y / ids.length;
    return acc;
  }, { x: 0, y: 0 });
}

function handScale(landmarks) {
  // Palm width + wrist-to-middle distance is stable enough for approach/retreat.
  return dist2(landmarks[5], landmarks[17]) + dist2(landmarks[0], landmarks[9]);
}

function pushHandHistory(tracker) {
  if (!tracker.history) tracker.history = [];
  const center = handCenter(tracker.landmarks);
  tracker.history.push({
    t: performance.now(),
    x: center.x,
    y: center.y,
    scale: handScale(tracker.landmarks),
  });
  if (tracker.history.length > HAND_HISTORY_FRAMES) tracker.history.shift();
}

function countDirectionChanges(values, minStep = 0.008) {
  let changes = 0;
  let lastDir = 0;
  for (let i = 1; i < values.length; i++) {
    const delta = values[i] - values[i - 1];
    if (Math.abs(delta) < minStep) continue;
    const dir = Math.sign(delta);
    if (lastDir && dir !== lastDir) changes++;
    lastDir = dir;
  }
  return changes;
}

function staticHandSignal(fingerStates, landmarks) {
  const up = Object.keys(FINGERS).filter(k => fingerStates[k]);
  const allUp = up.length === 5;
  const fist = up.length === 0;
  const thumbIndexDistance = dist2(landmarks[4], landmarks[8]);
  const palm = handScale(landmarks);

  if (allUp && isHandRaised(landmarks)) return { id: 'stop', label: 'Stop / High Five', kind: 'static' };
  if (allUp) return { id: 'open-palm', label: 'Open Palm', kind: 'static' };
  if (fist) return { id: 'fist', label: 'Fist', kind: 'static' };

  if (fingerStates.index && fingerStates.middle && !fingerStates.ring && !fingerStates.pinky) {
    return { id: 'peace', label: 'Peace / V Sign', kind: 'static' };
  }

  if (fingerStates.index && !fingerStates.middle && !fingerStates.ring && !fingerStates.pinky) {
    return { id: 'point', label: 'Pointing', kind: 'static' };
  }

  if (!fingerStates.index && fingerStates.middle && !fingerStates.ring && !fingerStates.pinky && !fingerStates.thumb) {
    return { id: 'middle-finger', label: 'Middle Finger', kind: 'static' };
  }

  if (fingerStates.thumb && !fingerStates.index && !fingerStates.middle && !fingerStates.ring && !fingerStates.pinky) {
    return { id: 'thumbs-up', label: 'Thumbs Up', kind: 'static' };
  }

  if (thumbIndexDistance < palm * 0.22 && fingerStates.middle && fingerStates.ring && fingerStates.pinky) {
    return { id: 'ok', label: 'OK Sign', kind: 'static' };
  }

  if (fingerStates.thumb && fingerStates.pinky && !fingerStates.index && !fingerStates.middle && !fingerStates.ring) {
    return { id: 'call-me', label: 'Call Me', kind: 'static' };
  }

  return { id: 'tracking', label: describeGesture(fingerStates), kind: 'static' };
}

function motionHandSignal(tracker) {
  const h = tracker.history || [];
  if (h.length < 10) return null;

  const recent = h.slice(-18);
  const xs = recent.map(p => p.x);
  const scales = recent.map(p => p.scale);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const firstScale = scales[0];
  const lastScale = scales[scales.length - 1];
  const scaleDelta = (lastScale - firstScale) / Math.max(firstScale, 0.001);
  const directionChanges = countDirectionChanges(xs);
  const allUp = Object.values(tracker.fingerStates || {}).filter(Boolean).length === 5;

  if (allUp && directionChanges >= 3 && maxX - minX > 0.08) {
    return { id: 'wave', label: 'Waving Hello', kind: 'motion' };
  }

  if (allUp && scaleDelta > 0.22) {
    return { id: 'come-closer', label: 'Come Closer', kind: 'motion' };
  }

  if (allUp && scaleDelta < -0.18) {
    return { id: 'go-away', label: 'Go Away', kind: 'motion' };
  }

  return null;
}

function updateHandSignal(tracker) {
  pushHandHistory(tracker);

  const motion = motionHandSignal(tracker);
  const stat = staticHandSignal(tracker.fingerStates, tracker.landmarks);
  const raw = motion || stat;

  if (!tracker.signal) {
    tracker.signal = { ...raw, streak: 1, pending: raw.id };
    return tracker.signal;
  }

  if (tracker.signal.id === raw.id) {
    tracker.signal = { ...raw, streak: 0, pending: raw.id };
    return tracker.signal;
  }

  const streak = tracker.signal.pending === raw.id ? tracker.signal.streak + 1 : 1;
  const needed = raw.kind === 'motion' ? 1 : 2;
  if (streak >= needed) {
    tracker.signal = { ...raw, streak: 0, pending: raw.id };
  } else {
    tracker.signal = { ...tracker.signal, streak, pending: raw.id };
  }

  return tracker.signal;
}

function frameSignature() {
  const w = detectCanvas.width;
  const h = detectCanvas.height;
  if (!w || !h) return 0;
  const pts = [
    detectCtx.getImageData(w >> 2, h >> 2, 1, 1).data,
    detectCtx.getImageData(w >> 1, h >> 2, 1, 1).data,
    detectCtx.getImageData((w * 3) >> 2, h >> 1, 1, 1).data,
    detectCtx.getImageData(w >> 1, (h * 3) >> 2, 1, 1).data,
  ];
  return pts.reduce((s, p) => s + p[0] + p[1] + p[2], 0);
}

function setStreamStatus(ok) {
  if (!streamStatusEl) return;
  streamStatusEl.textContent = ok ? 'LIVE' : 'FROZEN';
  streamStatusEl.className = ok ? 'stream-ok' : 'stream-bad';
}

function clearTrackingState() {
  currentHands = [];
  handTrackers = [];
  lastPairAssignment = [0, 1];
  particles = [];
  faceState = null;
  stableFaceExpression = null;
}

function processResults(results) {
  if (!results?.landmarks?.length) {
    for (const t of handTrackers) t.missed++;
    handTrackers = handTrackers.filter(t => t.missed <= 4);
    buildCurrentHands();
    return;
  }

  const detections = results.landmarks.map((raw, i) => ({
    raw,
    handedness: results.handednesses?.[i]?.[0]?.categoryName || 'Left',
  }));

  let alpha = SMOOTH_ALPHA;
  if (detections.length >= 2) {
    const separation = dist2(detections[0].raw[0], detections[1].raw[0]);
    if (separation < CLOSE_HANDS_DIST) alpha = SMOOTH_ALPHA_CLOSE;
  }

  assignDetections(detections, alpha);
  handTrackers = handTrackers.filter(t => t.missed <= 4);
  buildCurrentHands();
}

function resizeCanvases() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  viewCanvas.width = w;
  viewCanvas.height = h;
  canvas.width = w;
  canvas.height = h;
}

function drawVideoFrame() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return false;

  const cw = viewCanvas.width;
  const ch = viewCanvas.height;
  const videoAspect = vw / vh;
  const canvasAspect = cw / ch;
  let dw, dh, dx, dy;

  if (videoAspect > canvasAspect) {
    dh = ch;
    dw = ch * videoAspect;
    dx = (cw - dw) / 2;
    dy = 0;
  } else {
    dw = cw;
    dh = cw / videoAspect;
    dx = 0;
    dy = (ch - dh) / 2;
  }

  viewCtx.clearRect(0, 0, cw, ch);
  viewCtx.save();
  viewCtx.translate(dx + dw, dy);
  viewCtx.scale(-1, 1);
  viewCtx.drawImage(video, 0, 0, dw, dh);
  viewCtx.restore();

  if (detectCanvas.width !== vw || detectCanvas.height !== vh) {
    detectCanvas.width = vw;
    detectCanvas.height = vh;
  }
  detectCtx.drawImage(video, 0, 0, vw, vh);

  return true;
}

function landmarkToCanvas(lm) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const cw = canvas.width;
  const ch = canvas.height;
  if (!vw || !vh) return { x: (1 - lm.x) * cw, y: lm.y * ch };

  const videoAspect = vw / vh;
  const canvasAspect = cw / ch;
  let scale, offsetX, offsetY;

  if (videoAspect > canvasAspect) {
    scale = ch / vh;
    offsetX = (cw - vw * scale) / 2;
    offsetY = 0;
  } else {
    scale = cw / vw;
    offsetX = 0;
    offsetY = (ch - vh * scale) / 2;
  }

  return {
    x: (1 - lm.x) * vw * scale + offsetX,
    y: lm.y * vh * scale + offsetY,
  };
}

function drawHand(hand) {
  const { landmarks, fingerStates, color } = hand;
  const colors = color || HAND_COLORS.Left;
  const pts = landmarks.map(landmarkToCanvas);

  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  ctx.strokeStyle = colors.glow;
  ctx.lineWidth = 7;
  ctx.globalAlpha = 0.4;
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(pts[a].x, pts[a].y);
    ctx.lineTo(pts[b].x, pts[b].y);
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  for (const [a, b] of CONNECTIONS) {
    const grad = ctx.createLinearGradient(pts[a].x, pts[a].y, pts[b].x, pts[b].y);
    grad.addColorStop(0, colors.primary);
    grad.addColorStop(1, colors.secondary);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(pts[a].x, pts[a].y);
    ctx.lineTo(pts[b].x, pts[b].y);
    ctx.stroke();
  }

  for (let i = 0; i < pts.length; i++) {
    const isTip = TIP_INDICES.includes(i);
    const radius = isTip ? 6 : i === 0 ? 5 : 3.5;
    ctx.beginPath();
    ctx.arc(pts[i].x, pts[i].y, radius, 0, Math.PI * 2);
    ctx.fillStyle = isTip ? '#fff' : colors.primary;
    ctx.fill();
  }

  ctx.font = '600 11px Orbitron, sans-serif';
  ctx.textAlign = 'center';
  for (const [key, finger] of Object.entries(FINGERS)) {
    const tip = pts[finger.tip];
    ctx.fillStyle = fingerStates[key] ? '#39ff14' : 'rgba(255,255,255,0.35)';
    ctx.fillText(finger.name.toUpperCase(), tip.x, tip.y - 16);
  }
  ctx.restore();
}

function processFaceResults(results) {
  if (!results?.faceLandmarks?.length) {
    faceState = null;
    return;
  }

  const landmarks = results.faceLandmarks[0];
  const raw = classifyExpression(results.faceBlendshapes, landmarks);
  stableFaceExpression = stabilizeExpression(raw, stableFaceExpression, 3);

  faceState = {
    landmarks,
    expression: stableFaceExpression.expression,
    confidence: stableFaceExpression.confidence,
    meta: EXPRESSIONS[stableFaceExpression.expression],
  };
}

function drawFace() {
  if (!faceState?.landmarks) return;

  const pts = faceState.landmarks.map(landmarkToCanvas);
  const color = faceState.meta.color;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.55;
  ctx.shadowBlur = 0;

  ctx.beginPath();
  for (let i = 0; i < FACE_OVAL.length; i++) {
    const p = pts[FACE_OVAL[i]];
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  ctx.stroke();

  ctx.globalAlpha = 0.12;
  ctx.fillStyle = color;
  ctx.fill();

  // Subtle eye / mouth hints
  ctx.globalAlpha = 0.35;
  ctx.lineWidth = 1.5;
  for (const idx of [33, 133, 362, 263, 61, 291, 13, 14]) {
    const p = pts[idx];
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  ctx.restore();
}

function updateExpressionHUD() {
  if (!expressionPanel) return;

  if (!faceState) {
    expressionPanel.classList.add('hidden');
    hudCache.expression = '';
    return;
  }

  expressionPanel.classList.remove('hidden');
  const { meta, confidence, expression } = faceState;
  const confPct = `${Math.round(confidence * 100)}%`;

  if (hudCache.expression !== expression) {
    expressionEmoji.textContent = meta.emoji;
    expressionLabel.textContent = meta.label;
    expressionLabel.style.color = meta.color;
    expressionPanel.style.borderColor = `${meta.color}55`;
    expressionPanel.style.boxShadow = `0 0 24px ${meta.color}22`;
    hudCache.expression = expression;
  }

  expressionConfidence.textContent = confPct;
}
function updateHandHUD() {
  const count = currentHands.length;
  if (hudCache.handCount !== count) {
    handCountEl.textContent = `${count} hand${count !== 1 ? 's' : ''}`;
    hudCache.handCount = count;
  }

  const slots = TRACKER_COLORS.map((color, idx) => {
    const hand = currentHands.find(h => h.color?.label === color.label);
    return { color, hand, fallbackSide: idx === 0 ? 'Left' : 'Right' };
  });

  const panelsKey = slots.map(({ color, hand }) => {
    if (!hand) return `${color.label}:none`;
    const f = Object.keys(FINGERS).map(k => (hand.fingerStates[k] ? '1' : '0')).join('');
    return `${color.label}:${hand.handedness}${f}${hand.signal?.id || ''}`;
  }).join('|');

  if (hudCache.panelsKey === panelsKey) return;
  hudCache.panelsKey = panelsKey;

  for (const { color, hand, fallbackSide } of slots) {
    const panel = handPanelSlots[color.label];
    if (!panel) continue;
    const side = hand?.stableHandedness || hand?.handedness || fallbackSide;
    const signal = hand ? (hand.signal?.label || describeGesture(hand.fingerStates)) : 'No hand detected';
    const signalKind = hand?.signal?.kind === 'motion' ? 'motion' : 'static';

    panel.classList.toggle('inactive', !hand);
    panel.classList.toggle('left-hand', side === 'Left');
    panel.classList.toggle('right-hand', side === 'Right');

    const title = panel.querySelector('.hand-panel-title');
    if (title) title.textContent = `Hand ${color.label} · ${side}`;

    const badge = panel.querySelector('.signal-badge');
    if (badge) {
      badge.textContent = signal;
      badge.classList.toggle('motion', signalKind === 'motion');
      badge.classList.toggle('static', signalKind !== 'motion');
    }

    for (const key of Object.keys(FINGERS)) {
      const status = panel.querySelector(`[data-finger="${key}"] .finger-status`);
      if (!status) continue;
      const up = hand?.fingerStates?.[key] === true;
      status.textContent = up ? 'Extended' : 'Closed';
      status.classList.toggle('up', up);
      status.classList.toggle('down', !up);
    }
  }
}

function updateFooterLabel() {
  const parts = [];
  if (faceState) parts.push(`${faceState.meta.emoji} ${faceState.meta.label}`);
  if (currentHands.length) {
    parts.push(...currentHands.map(h => h.signal?.label || describeGesture(h.fingerStates)));
  }
  const text = parts.length
    ? parts.join(' · ')
    : 'Show your face or raise a hand';
  if (hudCache.gesture !== text) {
    gestureLabel.textContent = text;
    hudCache.gesture = text;
  }
}

function updateHUD() {
  updateHandHUD();
  updateExpressionHUD();
  updateFooterLabel();
}

function detectFrame() {
  if (!running || !handLandmarker || video.readyState < 2) return;

  if (video.paused) video.play().catch(() => {});

  if (!drawVideoFrame()) return;

  const sig = frameSignature();
  if (sig === lastFrameSig) {
    staleFrameCount++;
    setStreamStatus(staleFrameCount < 8);
    if (staleFrameCount >= 20) {
      restartCamera();
      return;
    }
  } else {
    staleFrameCount = 0;
    lastFrameSig = sig;
    setStreamStatus(true);
  }

  try {
    processResults(handLandmarker.detect(detectCanvas));

    if (faceLandmarker) {
      processFaceResults(faceLandmarker.detect(detectCanvas));
    }

    updateHUD();
    detectFrameCount++;
  } catch (err) {
    console.error('Detection error:', err);
    clearTrackingState();
  }
}

function renderOverlay() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawFace();
  for (const hand of currentHands) drawHand(hand);
}

function tickFps() {
  frameCount++;
  const now = performance.now();
  if (now - fpsLastTick >= 1000) {
    fps = frameCount;
    detectFps = detectFrameCount;
    frameCount = 0;
    detectFrameCount = 0;
    fpsLastTick = now;
    fpsEl.textContent = `${fps} FPS`;
    if (detectCountEl) detectCountEl.textContent = `${detectFps} det/s`;
  }
}

function paintLoop() {
  if (!running) return;
  tickFps();
  drawVideoFrame();
  renderOverlay();
  requestAnimationFrame(paintLoop);
}

function startTrackingLoops() {
  if (detectIntervalId) clearInterval(detectIntervalId);
  detectIntervalId = setInterval(detectFrame, 33);
  requestAnimationFrame(paintLoop);
}

function stopTrackingLoops() {
  running = false;
  if (detectIntervalId) {
    clearInterval(detectIntervalId);
    detectIntervalId = null;
  }
}

async function initModels() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'CPU',
    },
    runningMode: 'IMAGE',
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'CPU',
    },
    runningMode: 'IMAGE',
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
}

async function startCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
  }

  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'user',
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30 },
    },
    audio: false,
  });

  video.srcObject = mediaStream;
  video.playsInline = true;
  await video.play();

  await new Promise((resolve) => {
    if (video.videoWidth > 0) resolve();
    else video.addEventListener('loadeddata', resolve, { once: true });
  });

  lastFrameSig = -1;
  staleFrameCount = 0;
  stableFaceExpression = null;
  clearTrackingState();
}

async function restartCamera() {
  console.warn('Camera feed frozen — restarting stream');
  try {
    await startCamera();
  } catch (err) {
    console.error('Camera restart failed:', err);
    setStreamStatus(false);
  }
}

async function startApp() {
  loading.classList.remove('hidden');

  try {
    await startCamera();
    await initModels();

    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);

    startScreen.classList.remove('active');
    trackingScreen.classList.add('active');
    loading.classList.add('hidden');

    running = true;
    startTrackingLoops();
  } catch (err) {
    loading.classList.add('hidden');
    alert(
      err.name === 'NotAllowedError'
        ? 'Camera permission was denied. Please allow camera access and try again.'
        : `Failed to start: ${err.message}`
    );
  }
}

startBtn.addEventListener('click', startApp);

const browserHint = document.getElementById('browser-hint');
if (browserHint && isEmbeddedBrowser) {
  browserHint.classList.remove('hidden');
}
