# Contributing to GestureSense

Thank you for considering a contribution to GestureSense.

GestureSense is maintained by **Saad Kamal** and is intended to be approachable for people interested in browser-based computer vision, gesture interfaces, and interaction prototypes.

## Development Setup

No package install is required.

```bash
python3 -m http.server 8080
```

Open:

```text
http://localhost:8080
```

Use a normal browser such as Chrome, Safari, or Firefox for webcam testing.

## Contribution Guidelines

- Keep the app dependency-light. It should continue to run without a build step.
- Prefer readable heuristics over clever abstractions.
- Keep gesture and expression thresholds conservative to avoid noisy false positives.
- Test changes with:
  - No hands visible
  - One hand visible
  - Two hands close together
  - Hand entering/leaving frame
  - Neutral face
  - Strong expression or gesture
- Avoid adding code that sends webcam frames or landmarks to a remote service.

## Code Style

- Use plain modern JavaScript.
- Keep comments short and focused on why a heuristic exists.
- Prefer named helper functions for gesture or expression rules.
- Keep the HUD stable: update values/classes instead of rebuilding large DOM blocks.

## Good First Issues

- Improve gesture thresholds for different lighting/camera angles.
- Add a visual gesture history timeline.
- Add optional debug mode for raw landmark and blendshape values.
- Improve accessibility labels and keyboard affordances.
- Add screenshots or a short demo GIF to the README.

## Reporting Bugs

When reporting a bug, please include:

- Browser and version
- Operating system
- Whether you used a normal browser or an embedded webview
- What gesture/expression you expected
- What GestureSense detected instead
- Whether `LIVE` and `det/s` were updating in the HUD

## Project Owner

Primary contributor and maintainer: **Saad Kamal**
