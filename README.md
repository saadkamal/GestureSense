# GestureSense

**GestureSense** is a browser-based computer-vision demo for real-time hand tracking, gesture recognition, and facial-expression detection.

It uses the webcam, MediaPipe Tasks Vision, and HTML canvas overlays to create a polished gesture-control prototype that can recognize hands, fingers, common hand signals, motion gestures, and facial expressions directly in the browser.

Created and maintained by **Saad Kamal**.

## Features

- Real-time webcam preview with a full-screen mirrored canvas view
- Two-hand tracking with persistent Hand A / Hand B HUD panels
- Per-finger state detection for thumb, index, middle, ring, and pinky
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
