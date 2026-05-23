# Security Policy

GestureSense is maintained by **Saad Kamal**.

## Supported Versions

This project is currently a browser-based demo. The latest version in the main project directory is the supported version.

## Privacy and Camera Data

GestureSense processes webcam frames locally in the browser using MediaPipe Tasks Vision. The app does not intentionally send webcam frames, landmarks, or expression data to a remote server.

If you add features that export or transmit camera-derived data, document that behavior clearly and require explicit user consent.

## Reporting a Vulnerability

If you discover a security or privacy issue:

1. Do not publish exploit details publicly before the maintainer has a chance to respond.
2. Contact **Saad Kamal**, the project maintainer.
3. Include steps to reproduce and explain the possible impact.

## Security Expectations for Contributors

- Do not add third-party scripts unless they are necessary and documented.
- Prefer local browser processing for webcam data.
- Do not collect or persist camera data by default.
- Avoid hidden network requests.
- Keep user consent visible and explicit for camera access.
