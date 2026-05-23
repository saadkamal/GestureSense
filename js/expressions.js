/**
 * GestureSense expression classifier.
 *
 * Created and maintained by Saad Kamal.
 *
 * MediaPipe FaceLandmarker exposes blendshapes and landmarks, not guaranteed
 * emotion labels. These rules intentionally favor neutral unless a signal is
 * clear enough for a demo-quality expression classification.
 */
export const EXPRESSIONS = {
  happy:     { emoji: '😊', label: 'Happy',     color: '#ffd93d' },
  slightHappy: { emoji: '🙂', label: 'Slight Smile', color: '#ffe680' },
  sad:       { emoji: '😢', label: 'Sad',       color: '#6b9eff' },
  angry:     { emoji: '😠', label: 'Angry',     color: '#ff4757' },
  surprised: { emoji: '😲', label: 'Surprised', color: '#ffa502' },
  disgusted: { emoji: '🤢', label: 'Disgusted', color: '#78e08f' },
  fearful:   { emoji: '😨', label: 'Fearful',   color: '#c8b6ff' },
  bored:      { emoji: '😑', label: 'Bored',      color: '#9aa4b2' },
  sleepy:     { emoji: '😴', label: 'Sleepy',     color: '#b8a6ff' },
  neutral:   { emoji: '😐', label: 'Neutral',   color: '#8899aa' },
};

export const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
  379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
  234, 127, 162, 21, 54, 103, 67, 109, 10,
];

export function blendshapeMap(blendshapes) {
  const b = {};
  if (!blendshapes?.length) return b;
  for (const s of blendshapes[0].categories) {
    if (s.categoryName) b[s.categoryName] = s.score;
  }
  return b;
}

export function avg(b, keys) {
  let sum = 0;
  let n = 0;
  for (const k of keys) {
    if (b[k] != null) { sum += b[k]; n++; }
  }
  return n ? sum / n : 0;
}

function v(b, key) {
  return b[key] || 0;
}

const SAD_BASELINE_FRAMES = 8;
const sadBaseline = {
  samples: [],
  value: null,
};

function mouthCornerGeometry(landmarks) {
  if (!landmarks?.length || landmarks.length < 292) return null;

  const leftCorner = landmarks[61];
  const rightCorner = landmarks[291];
  const upperLip = landmarks[13];
  const lowerLip = landmarks[14];
  const nose = landmarks[1];
  const chin = landmarks[152];
  const forehead = landmarks[10];

  const faceHeight = Math.max(chin.y - nose.y, 0.05);
  const upperFace = Math.max(nose.y - forehead.y, 0.02);
  const leftDrop = (leftCorner.y - upperLip.y) / faceHeight;
  const rightDrop = (rightCorner.y - upperLip.y) / faceHeight;
  const avgDrop = (leftDrop + rightDrop) / 2;
  const lipCenterY = (upperLip.y + lowerLip.y) / 2;

  return {
    leftDrop,
    rightDrop,
    avgDrop,
    // Useful fallback: corners relative to mouth center.
    centerDrop: ((leftCorner.y + rightCorner.y) / 2 - lipCenterY) / faceHeight,
    // Simple head-pitch proxy. Tilting up/down changes this and can fake a frown.
    pitch: upperFace / faceHeight,
  };
}

function sadGeometryScore(landmarks, smile, jawOpen, frown, mouthLowerDown, mouthShrugLower) {
  const geom = mouthCornerGeometry(landmarks);
  if (geom == null) return 0;

  // Tongue-out / open-mouth shapes distort lip corners and look like a frown in 2D.
  if (jawOpen > 0.16 || mouthLowerDown > 0.18) return 0;

  // Silently learn neutral mouth position while the face is relaxed.
  if (sadBaseline.value == null && smile < 0.28 && jawOpen < 0.28) {
    sadBaseline.samples.push(geom);
    if (sadBaseline.samples.length >= SAD_BASELINE_FRAMES) {
      sadBaseline.value = {
        leftDrop: sadBaseline.samples.reduce((s, x) => s + x.leftDrop, 0) / sadBaseline.samples.length,
        rightDrop: sadBaseline.samples.reduce((s, x) => s + x.rightDrop, 0) / sadBaseline.samples.length,
        avgDrop: sadBaseline.samples.reduce((s, x) => s + x.avgDrop, 0) / sadBaseline.samples.length,
        centerDrop: sadBaseline.samples.reduce((s, x) => s + x.centerDrop, 0) / sadBaseline.samples.length,
        pitch: sadBaseline.samples.reduce((s, x) => s + x.pitch, 0) / sadBaseline.samples.length,
      };
    }
    return 0;
  }

  if (sadBaseline.value == null) return 0;

  const pitchDelta = Math.abs(geom.pitch - sadBaseline.value.pitch);
  const hasActiveSadMouth = frown > 0.07 || mouthLowerDown > 0.05 || mouthShrugLower > 0.05;

  // Head tilt changes projected lip-corner positions. If the head angle changed
  // but the mouth itself is not actively frowning, treat it as pose, not sadness.
  if (pitchDelta > 0.025 && !hasActiveSadMouth) return 0;
  if (pitchDelta > 0.06) return 0;

  const leftDelta = geom.leftDrop - sadBaseline.value.leftDrop;
  const rightDelta = geom.rightDrop - sadBaseline.value.rightDrop;
  const avgDelta = geom.avgDrop - sadBaseline.value.avgDrop;
  const centerDelta = geom.centerDrop - sadBaseline.value.centerDrop;

  // Both mouth corners must move down. Low threshold because this is normalized by face height.
  const bothCornersDown = leftDelta > 0.0015 && rightDelta > 0.0015;
  if (!bothCornersDown) return 0;

  let score = Math.max(avgDelta * 95, centerDelta * 80, 0);

  // Even with an active frown, reduce geometry confidence during head movement.
  if (pitchDelta > 0.025) score *= 0.45;

  return Math.min(score, 1);
}

/**
 * Conservative expression classifier.
 *
 * MediaPipe face blendshapes are action-unit-like signals, not emotion labels.
 * To avoid false positives, each non-neutral expression needs multiple pieces
 * of evidence. Weak or ambiguous expressions intentionally remain neutral.
 */
export function classifyExpression(blendshapes, landmarks) {
  const b = blendshapeMap(blendshapes);

  if (!Object.keys(b).length) {
    return {
      expression: 'neutral',
      confidence: 0,
      meta: EXPRESSIONS.neutral,
    };
  }

  const smile = avg(b, ['mouthSmileLeft', 'mouthSmileRight']);
  const frown = avg(b, ['mouthFrownLeft', 'mouthFrownRight']);
  const browDown = avg(b, ['browDownLeft', 'browDownRight']);
  const browInnerUp = v(b, 'browInnerUp');
  const browOuterUp = avg(b, ['browOuterUpLeft', 'browOuterUpRight']);
  const jawOpen = v(b, 'jawOpen');
  const eyeWide = avg(b, ['eyeWideLeft', 'eyeWideRight']);
  const eyeSquint = avg(b, ['eyeSquintLeft', 'eyeSquintRight']);
  const noseSneer = avg(b, ['noseSneerLeft', 'noseSneerRight']);
  const mouthPress = v(b, 'mouthPress');
  const cheekSquint = avg(b, ['cheekSquintLeft', 'cheekSquintRight']);
  const mouthPucker = v(b, 'mouthPucker');
  const mouthLowerDown = avg(b, ['mouthLowerDownLeft', 'mouthLowerDownRight']);
  const mouthShrugLower = avg(b, ['mouthShrugLowerLeft', 'mouthShrugLowerRight']);
  const eyeLookDown = avg(b, ['eyeLookDownLeft', 'eyeLookDownRight']);
  const sadGeom = sadGeometryScore(landmarks, smile, jawOpen, frown, mouthLowerDown, mouthShrugLower);

  const candidates = [];

  function addCandidate(expression, score, evidence, blockers = false, minScore = 0.35) {
    if (!blockers && evidence >= 2 && score >= minScore) {
      candidates.push({ expression, score, evidence });
    }
  }

  addCandidate(
    'happy',
    smile * 0.9 + cheekSquint * 0.25,
    (smile > 0.32 ? 1 : 0) + (cheekSquint > 0.12 ? 1 : 0) + (frown < 0.12 ? 1 : 0),
    frown > 0.2 || browDown > 0.3,
  );

  addCandidate(
    'slightHappy',
    smile * 0.75 + cheekSquint * 0.15,
    (smile > 0.16 ? 1 : 0) + (smile < 0.36 ? 1 : 0) + (frown < 0.1 ? 1 : 0) + (browDown < 0.18 ? 1 : 0),
    smile > 0.42 || frown > 0.16 || browDown > 0.22 || jawOpen > 0.22,
    0.14,
  );

  addCandidate(
    'surprised',
    jawOpen * 0.75 + browOuterUp * 0.45 + eyeWide * 0.3,
    (jawOpen > 0.14 ? 1 : 0) + (browOuterUp > 0.14 ? 1 : 0) + (eyeWide > 0.08 ? 1 : 0),
    smile > 0.35 || browDown > 0.3,
    0.22,
  );

  addCandidate(
    'angry',
    browDown * 0.75 + eyeSquint * 0.25 + mouthPress * 0.25 + noseSneer * 0.2,
    (browDown > 0.3 ? 1 : 0) + (eyeSquint > 0.16 ? 1 : 0) + (mouthPress > 0.12 ? 1 : 0) + (noseSneer > 0.1 ? 1 : 0),
    smile > 0.18 || browInnerUp > 0.35,
  );

  addCandidate(
    'sad',
    Math.max(
      frown * 2.2 + browInnerUp * 0.8 + mouthLowerDown * 1.1 + mouthShrugLower * 1.0,
      sadGeom * 2.0,
    ),
    (sadGeom > 0.035 ? 1 : 0)
      + (frown > 0.07 ? 1 : 0)
      + (browInnerUp > 0.08 ? 1 : 0)
      + (mouthLowerDown > 0.04 || mouthShrugLower > 0.04 ? 1 : 0)
      + (smile < 0.22 ? 1 : 0),
    smile > 0.28 || browDown > 0.36 || jawOpen > 0.16 || mouthLowerDown > 0.18,
    0.08,
  );

  addCandidate(
    'disgusted',
    noseSneer * 0.75 + mouthPucker * 0.2,
    (noseSneer > 0.28 ? 1 : 0) + (smile < 0.15 ? 1 : 0),
    smile > 0.2,
  );

  addCandidate(
    'fearful',
    eyeWide * 0.45 + browInnerUp * 0.4 + jawOpen * 0.25,
    (eyeWide > 0.2 ? 1 : 0) + (browInnerUp > 0.22 ? 1 : 0) + (jawOpen > 0.12 ? 1 : 0),
    browDown > 0.25 || smile > 0.25,
  );

  addCandidate(
    'bored',
    mouthPress * 0.75 + eyeLookDown * 0.45 + eyeSquint * 0.25,
    (mouthPress > 0.2 ? 1 : 0)
      + (eyeLookDown > 0.28 ? 1 : 0)
      + (eyeSquint > 0.22 ? 1 : 0)
      + (smile < 0.08 ? 1 : 0)
      + (jawOpen < 0.08 ? 1 : 0),
    smile > 0.12 || jawOpen > 0.1 || sadGeom > 0.06 || frown > 0.12 || browOuterUp > 0.12,
    0.42,
  );

  addCandidate(
    'sleepy',
    eyeSquint * 0.85 + eyeLookDown * 0.45 + jawOpen * 0.35,
    (eyeSquint > 0.32 ? 1 : 0)
      + (eyeLookDown > 0.22 ? 1 : 0)
      + (jawOpen > 0.12 || mouthPress > 0.16 ? 1 : 0)
      + (smile < 0.1 ? 1 : 0),
    smile > 0.14 || browOuterUp > 0.16 || browDown > 0.25 || sadGeom > 0.08,
    0.46,
  );

  candidates.sort((a, b) => b.score - a.score);
  const best = candidates[0];

  if (best) {
    return {
      expression: best.expression,
      confidence: Math.min(Math.max(best.score, 0.35), 1),
      meta: EXPRESSIONS[best.expression],
    };
  }

  return { expression: 'neutral', confidence: 0.75, meta: EXPRESSIONS.neutral };
}

export function stabilizeExpression(raw, prev, streak = 3) {
  if (!prev) {
    return { expression: raw.expression, confidence: raw.confidence, streak: 0, pending: null };
  }

  // Same expression — keep it
  if (raw.expression === prev.expression) {
    return {
      expression: prev.expression,
      confidence: prev.confidence * 0.7 + raw.confidence * 0.3,
      streak: 0,
      pending: null,
    };
  }

  const needed = raw.expression === 'neutral'
    ? 2
    : raw.expression === 'happy' || raw.expression === 'surprised'
      ? 2
      : streak;

  const pending = raw.expression;
  const nextStreak = prev.pending === pending ? (prev.streak || 0) + 1 : 1;

  if (nextStreak >= needed) {
    return { expression: pending, confidence: raw.confidence, streak: 0, pending: null };
  }

  return {
    expression: prev.expression,
    confidence: prev.confidence,
    streak: nextStreak,
    pending,
  };
}
