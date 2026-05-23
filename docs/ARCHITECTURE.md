# GestureSense Architecture

GestureSense is a lightweight browser app created and maintained by **Saad Kamal**.

The project intentionally avoids a build pipeline so contributors can run it with any static file server.

## Runtime Pipeline

```text
Webcam stream
  -> hidden video element
  -> visible canvas (`#view`)
  -> hidden detection canvas
  -> MediaPipe HandLandmarker + FaceLandmarker
  -> normalized landmarks / blendshapes
  -> gesture + expression classifiers
  -> overlay canvas + HUD updates
```

## Why Canvas Instead of Direct Video

The app draws the webcam feed into a canvas instead of showing the video element directly. This gives us:

- Consistent mirroring
- Stable object-fit behavior
- A reliable source for MediaPipe image-mode detection
- Better behavior in browsers or embedded webviews where video frame callbacks can be unreliable

## Main Files

### `index.html`

Defines the app shell:

- Start screen
- Hidden webcam video
- Visible webcam canvas
- Transparent overlay canvas
- Persistent HUD panels

The hand HUD uses fixed Hand A / Hand B elements so the layout does not jump when hands enter or leave the frame.

### `css/styles.css`

Owns the visual system:

- Full-screen cyberpunk aesthetic
- HUD layout
- Expression panel styling
- Persistent hand panels
- Motion/static signal badges

### `js/app.js`

Owns runtime behavior:

- Camera startup and restart handling
- MediaPipe model loading
- Hand tracker identity matching
- Finger state detection
- Static and motion gesture classification
- Face landmark rendering
- HUD updates

### `js/expressions.js`

Owns expression classification:

- Expression metadata
- Face oval landmark indices
- Blendshape helpers
- Conservative expression rules
- Stabilization/debounce between expression states

## Hand Tracking Design

MediaPipe returns hand detections per frame, but the order can change when two hands overlap. GestureSense keeps persistent hand trackers:

- `Hand A` / `Hand B` stay visually stable
- Trackers match detections by wrist position and landmark-cloud distance
- Handedness labels are debounced before changing
- Finger states are debounced to reduce flicker

## Gesture Detection Design

GestureSense recognizes two categories:

### Static Signals

Static signals are based on which fingers are extended:

- Fist
- Open palm
- Stop / high five
- Pointing
- Peace sign
- Thumbs up
- OK sign
- Call me
- Middle finger

### Motion Signals

Motion signals use short hand-history buffers:

- Wave: open palm with repeated horizontal direction changes
- Come closer: open palm with growing hand scale
- Go away: open palm with shrinking hand scale

These are heuristics, not a formal sign-language model.

## Expression Detection Design

Face expressions use MediaPipe blendshapes. The classifier intentionally favors neutral unless the signal is clear, because blendshapes are action units and can be noisy across faces, lighting, and camera angles.

Some expressions are necessarily approximate. For example, sadness is difficult from webcam-only blendshapes, so the implementation combines frown-related blendshapes with mouth-corner geometry.

## Known Limitations

- Expressions vary significantly between people and cameras.
- Tongue-out, strong head tilt, and partial occlusion can distort mouth geometry.
- Motion gestures need adequate lighting and a stable frame rate.
- Embedded webviews may freeze or degrade webcam frames.

## Future Improvements

- Add a debug mode for raw landmark/blendshape values.
- Add optional per-user calibration for expression thresholds.
- Add a settings panel for gesture sensitivity.
- Add test fixtures with recorded landmark sequences.
- Add demo screenshots and videos to the README.
