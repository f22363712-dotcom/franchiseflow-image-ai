# FranchiseFlow

**One Moving Storefront Video → Three Brand Branch Videos via Planar Vision AI & Geometric Tracking**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FranchiseFlow solves a core content bottleneck for franchise retail and multi-branch brands: turning a single moving storefront video clip into multiple location-specific branch variants.

Instead of expensive and jittery frame-by-frame multimodal API calls, FranchiseFlow implements an intelligent hybrid pipeline:
1. **Physical Plane Localization (Vision AI / Calibration)**: The first frame is passed to **gemini-3.1-flash-image** for initial plane discovery, paired with an explicit **Manual Calibration Baseline** (`quad.json`) when 3D perspective angles require ground-truth alignment.
2. **Sub-pixel Geometric Tracking (Local CV)**: OpenCV SIFT feature matching with RANSAC homography estimation tracks the moving sign through camera motion, backed by Lucas-Kanade (LK) optical flow.
3. **Multi-Branch Synchronous Compositing**: Exactly three branch brand assets are perspective-warped and composited onto the original video simultaneously, generating three web-compatible H.264 MP4 files.

---

## 🎯 Architecture

```
[Moving Storefront MP4] (Frame 0)
        │
        ▼
[gemini-3.1-flash-image / Calibration] ──► [Planar Quad (TL, TR, BR, BL)]
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
- Toggle between **gemini-3.1-flash-image 视觉定位 (初筛测试)** and **人工校准基线 (Manual Calibration)**.
- Click **载入默认样例素材** to instantly review the verified H.264 outputs and telemetry metrics.

### 3. Generate Demo Cut & Run CLI Pipeline

Generate the clean 6.0-second (144-frame) demo cut from the storefront video:

```bash
ffmpeg -y -i spikes/franchiseflow/fixtures/real_storefront.mp4 -t 6.0 -c copy spikes/franchiseflow/fixtures/demo_cut.mp4
```

Run the deterministic tracking and H.264 transcode pipeline via CLI:

```bash
python spikes/franchiseflow/tracker.py \
  --video spikes/franchiseflow/fixtures/demo_cut.mp4 \
  --quad-json spikes/franchiseflow/quad.json \
  --signs spikes/franchiseflow/assets/riverside.png spikes/franchiseflow/assets/downtown.png spikes/franchiseflow/assets/northpark.png \
  --out-dir spikes/franchiseflow/outputs_demo_h264
```

Outputs are written to `outputs_demo_h264/`:
- `branch_1_riverside.mp4` (H.264, yuv420p, faststart, browser-playable)
- `branch_2_downtown.mp4` (H.264, yuv420p, faststart, browser-playable)
- `branch_3_northpark.mp4` (H.264, yuv420p, faststart, browser-playable)
- `tracking.jsonl` (per-frame telemetry)
- `summary.json` (reprojection residuals, inlier counts, processing time)

---

## 🤖 Vision Model Configuration & Positioning

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

Configuration variables (read strictly from environment or `.env`):
- `BANANA_API_KEY`: API key (never hardcoded or committed)
- `BANANA_API_BASE_URL`: Gateway URL (defaults to `https://keungliang.dpdns.org/v1`)
- `BANANA_VISION_MODEL`: Model identifier (defaults to `gemini-3.1-flash-image`)

Test first-frame sign localization:

```bash
python spikes/franchiseflow/auto_quad.py \
  --image spikes/franchiseflow/fixtures/real_storefront_first.jpg \
  --out-json spikes/franchiseflow/quad_auto.json \
  --preview spikes/franchiseflow/quad_auto_preview.jpg \
  --target-hint "Dark gray storefront signage board displaying 'STATE OF CONFUSION'"
```

> **Engineering Note on Localization**: In evaluations with `gemini-3.1-flash-image`, current multimodal vision models lack fine-grained 3D perspective orientation on oblique commercial facades and frequently predict quadrilaterals cutting across brick walls and windows. FranchiseFlow explicitly transparentizes this limitation: the model provides initial candidate screening, while the **Manual Calibration Baseline** (`quad.json`) provides verified geometric corners. Passing `--quad-json` does NOT imply automated model discovery.

---

## 🧪 Testing & Verification

Run the entire test suite (22 tests covering HTTP contracts, error redaction, synthetic motion, FFmpeg missing checks, and Web API streaming):

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

- **Residual Definition**: Reported reprojection residuals (e.g. 0.30 px – 0.36 px) are **internal SIFT inlier fit residuals** from RANSAC homography estimation, **not ground-truth sign boundary errors**.
- **Short-test Metrics (20 frames)**: 475 SIFT features extracted in frame 0, median 288 inliers, 0 failed frames, median reprojection residual 0.30 px.
- **Full Demo Cut (144 frames, 6.0s, 24 fps)**:
  - Initialization: Manual calibration baseline (`quad.json`)
  - Total processing time: 122s (including H.264 libx264 yuv420p faststart transcoding; pure tracking ~63s)
  - First-frame SIFT features: 286
  - Median inliers: 113 consensus points
  - Median reprojection residual: 0.362 px (P95: 1.86 px)
  - Failed frames: 0 (143/143 tracked via anchor SIFT)
- **Scope & Physical Boundaries**:
  - The fallback policy only handles *detected tracking failures* (e.g. low inlier counts or degenerate homographies) by preserving original frames.
  - The system currently has **no foreground occlusion segmentation** (e.g. SAM or video matting) and does **not preserve original audio**.

---

## 🏆 Banana Hacks Submission Checklist

- [x] **Project Name**: FranchiseFlow
- [x] **Public GitHub Repository**: [franchiseflow-image-ai](https://github.com/f22363712-dotcom/franchiseflow-image-ai)
- [x] **Demo Video Script**: Under 3 minutes (see [docs/FranchiseFlow_Demo脚本.md](docs/FranchiseFlow_Demo脚本.md))
- [x] **Strict Secret Hygiene**: Zero hardcoded keys or leaked secrets
- [x] **Hackathon Window Creation**: Developed during the official hackathon window
- [x] **Credits & Third-Party Resources**:
  - Storefront video fixture from Pexels (License record in `spikes/franchiseflow/fixtures/source.md`)
  - Vision Model: `gemini-3.1-flash-image` (via authorized endpoint)
  - Core Libraries: OpenCV, NumPy, Pillow, Flask, FFmpeg
