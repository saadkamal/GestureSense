# GestureSense

**GestureSense** is a browser-based computer-vision demo for real-time **per-finger hand tracking**, gesture recognition, and facial-expression detection.

Raise one or both hands in front of your webcam and see **each finger detected individually** — thumb, index, middle, ring, and pinky — with live open/closed status on the HUD and color-coded skeleton overlays. GestureSense also recognizes common hand signals, motion gestures, and facial expressions, all running in the browser via MediaPipe Tasks Vision.

Created and maintained by **Saad Kamal**.

## Live Demo

Try GestureSense in your browser:

**https://gesturesense-production.up.railway.app/**

Click **Enable Camera** and allow webcam access. Raise your hand(s) and watch each finger update in real time.

## Individual Finger Detection

GestureSense tracks **all five fingers on each hand** separately:

| Finger | What you see |
|--------|----------------|
| **Thumb** | Extended or tucked (works from front and side views) |
| **Index** | Raised or closed |
| **Middle** | Raised or closed |
| **Ring** | Raised or closed |
| **Pinky** | Raised or closed |

- **Two hands** — Hand A and Hand B panels stay on screen with per-finger rows
- **Visual overlay** — Landmarks and finger segments highlight on the video feed
- **Stable HUD** — Finger labels stay fixed; only the status text changes (e.g. `Up` / `Closed`)

This is the core of the demo: proof that gesture-control input can be built finger-by-finger, not just as a single “hand present” blob.

## Features

- Real-time webcam preview with a full-screen mirrored canvas view
- Two-hand tracking with persistent Hand A / Hand B HUD panels
- **Per-finger detection:** thumb, index, middle, ring, and pinky (open/closed per hand)
- Hand signals:
  - Open palm
  - Stop / high five
  - Fist
  - Pointing
  - Peace / V sign
  - Thumbs up
  - OK sign
  - Call me
  - Middle finger
- Motion gestures:
  - Waving hello
  - Come closer
  - Go away
- Facial-expression detection:
  - Neutral
  - Slight smile
  - Happy
  - Sad
  - Angry
  - Surprised
  - Disgusted
  - Fearful
  - Bored
  - Sleepy
- Cyberpunk-style HUD with live FPS, detection rate, stream status, expression panel, and hand panels
- Webcam freeze detection and automatic camera restart attempt

## Demo Requirements

GestureSense runs in the browser and does not require a build step.

Recommended browsers:

- Google Chrome
- Safari
- Firefox

Cursor's built-in browser and other Electron webviews may have limited webcam support. For best results, run the demo in a normal browser.

## Quick Start

From the project root:

```bash
python3 -m http.server 8080
```

Then open:

```text
http://localhost:8080
```

Click **Enable Camera** and allow webcam permission.

## Project Structure

```text
.
├── index.html              # App shell and persistent HUD markup
├── css/
│   └── styles.css          # Visual system and HUD styling
├── js/
│   ├── app.js              # Webcam, MediaPipe models, hand tracking, gestures, rendering
│   └── expressions.js      # Facial-expression classification helpers
├── docs/
│   └── ARCHITECTURE.md     # Implementation notes
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── LICENSE
└── README.md
```

## How It Works

GestureSense uses two MediaPipe Tasks Vision models:

- `HandLandmarker` for hand landmarks and handedness
- `FaceLandmarker` for facial landmarks and blendshapes

The app draws the webcam stream into a canvas, runs detection on a hidden detection canvas, and renders hand/face overlays on a separate transparent canvas. This keeps the webcam feed, ML input, and visual overlay separate and easier to maintain.

## Notes on Gesture Detection

Gesture recognition is intentionally heuristic. Hand tracking models provide landmarks, not semantic gestures. GestureSense derives signals from:

- Finger extension states
- Palm scale and movement history
- Persistent hand IDs
- Landmark distances and motion direction changes

The current goal is a strong demo foundation, not a production-grade sign-language interpreter.

## Privacy

GestureSense runs locally in your browser. Webcam frames are processed in the browser by MediaPipe Tasks Vision. The app does not upload camera data to a server.

## Contributor

Primary contributor and project owner:

**Saad Kamal**

## License

This project is licensed under the MIT License. See `LICENSE`.
