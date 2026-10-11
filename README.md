# FranchiseFlow

**One Moving Storefront Video → Three Brand Branch Videos via Planar Vision AI & Geometric Tracking**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FranchiseFlow solves a core content bottleneck for franchise retail and multi-branch brands: turning a single moving storefront video clip into multiple location-specific branch variants.

Instead of expensive and jittery frame-by-frame multimodal API calls, FranchiseFlow implements an intelligent hybrid pipeline:
1. **Physical Plane Localization (Vision AI)**: The first frame is sent once to **GLM-5.3-flash** to identify the physical four-corner planar quadrilateral of the storefront signage.
2. **Sub-pixel Geometric Tracking (Local CV)**: OpenCV SIFT feature matching with RANSAC homography estimation tracks the moving sign through camera motion, backed by Lucas-Kanade (LK) optical flow.
3. **Multi-Branch Synchronous Compositing**: Exactly three branch brand assets are perspective-warped and composited onto the original video simultaneously, generating three broadcast-ready MP4 files.

---

## 🎯 Architecture

```
[Moving Storefront MP4] (Frame 0)
        │
        ▼
[GLM-5.3-flash Vision AI] ───► [Planar Quad (TL, TR, BR, BL)]
        │                                  │
        └─────────────────┬────────────────┘
                          ▼
             [OpenCV Planar Tracker]
         (SIFT Anchor Matching + RANSAC)
               (LK Flow Fallback)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    [Branch 1 MP4]  [Branch 2 MP4]  [Branch 3 MP4]
    (Riverside St)   (Downtown Ave)   (Northpark Mall)
```

---

## 🚀 Quickstart

### 1. Requirements

- Python 3.11+
- Dependencies: `opencv-python`, `numpy`, `pillow`, `flask`

```bash
pip install opencv-python numpy pillow flask
```

### 2. Run the Web Demo UI

Start the local demo interface:

```bash
python prototype/franchiseflow/app.py
```

Open `http://127.0.0.1:5000/` in your browser:
- Upload any moving storefront clip (or use `spikes/franchiseflow/fixtures/real_storefront.mp4`).
- Upload three branch sign images (or use the sample branch assets in `spikes/franchiseflow/assets/`).
- Toggle between **GLM-5.3-flash Vision AI** and **Manual Fallback**.
- View detected planar corners, real-time tracking metrics, and play/download all three output videos.

### 3. Run the CLI Pipeline

You can also run the deterministic tracking pipeline via CLI:

```bash
python spikes/franchiseflow/tracker.py \
  --video spikes/franchiseflow/fixtures/real_storefront.mp4 \
  --quad spikes/franchiseflow/quad.json \
  --signs spikes/franchiseflow/assets/riverside.png spikes/franchiseflow/assets/downtown.png spikes/franchiseflow/assets/northpark.png \
  --out-dir spikes/franchiseflow/outputs2
```

Outputs are written to `outputs2/`:
- `branch_1_riverside.mp4`
- `branch_2_downtown.mp4`
- `branch_3_northpark.mp4`
- `tracking.jsonl` (per-frame telemetry)
- `summary.json` (reprojection errors, inlier counts, failed frames)

---

## 🤖 Vision Model Configuration

Copy `.env.example` to `.env` and set your gateway credentials:

```bash
cp .env.example .env
```

Configuration variables (read strictly from environment or `.env`):
- `BANANA_API_KEY`: API key (never hardcoded or committed)
- `BANANA_API_BASE_URL`: Gateway URL (defaults to `https://e.c.h-e.top/v1`)
- `BANANA_VISION_MODEL`: Model identifier (defaults to `GLM-5.3-flash`)

Test first-frame sign localization:

```bash
python spikes/franchiseflow/auto_quad.py \
  --image spikes/franchiseflow/fixtures/real_storefront_first.jpg \
  --out-json spikes/franchiseflow/quad_auto.json \
  --preview spikes/franchiseflow/quad_auto_preview.jpg
```

> **Gateway Access Note**: During testing, public gateway access returned HTTP 403. The system gracefully surfaces this status and provides an explicit manual engineering fallback (`--quad`) so that downstream tracking and video generation remain fully functional offline without faking model outputs. See [docs/FranchiseFlow_API接入记录.md](docs/FranchiseFlow_API接入记录.md).

---

## 🧪 Testing & Verification

Run the entire test suite (20 tests covering HTTP contracts, error redaction, synthetic motion, failed frame safety, and Web API routes):

```bash
# Tracker & Vision Client tests
python -m unittest discover -s spikes/franchiseflow/tests -v

# Prototype Web API tests
python -m unittest discover -s prototype/franchiseflow/tests -v

# Code cleanliness & secret leak scan
git diff --check
git grep -n -I -E 'sk-[A-Za-z0-9]{20,}|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}' -- ':!D*'
```

---

## 📊 Measured Performance & Metrics

Tested on 1280×720, 24 fps, 233-frame moving storefront video (`real_storefront.mp4`):
- **SIFT Inliers (Median)**: 123 consensus feature points
- **Reprojection Residuals (Median)**: 0.21 – 0.31 px
- **Reprojection Residuals (P95)**: 1.21 – 1.70 px
- **Failure Frame Policy**: On frames with severe foreground occlusion (e.g., pedestrian crossing sign at frames 155–161), the system preserves the original frame rather than rendering a frozen/floating artifact.
- **Demo Cut (`demo_cut.mp4`)**: A 144-frame (6.0s) clean unoccluded pan cut achieves 100% stable tracking.

---

## 🏆 Banana Hacks Submission Checklist

- [x] **Project Name**: FranchiseFlow
- [x] **Public GitHub Repository**: [franchiseflow-image-ai](https://github.com/f22363712-dotcom/franchiseflow-image-ai)
- [x] **Demo Video Script**: Under 3 minutes (see [docs/FranchiseFlow_Demo脚本.md](docs/FranchiseFlow_Demo脚本.md))
- [x] **Strict Secret Hygiene**: Zero hardcoded keys or leaked secrets
- [x] **Hackathon Window Creation**: Developed during the official hackathon window
- [x] **Credits & Third-Party Resources**:
  - Storefront video fixture from Pexels (License record in `spikes/franchiseflow/fixtures/source.md`)
  - Vision Model: GLM-5.3-flash
  - Core Libraries: OpenCV, NumPy, Pillow, Flask
