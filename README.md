# FranchiseFlow

FranchiseFlow turns one moving storefront video into multiple branch-specific videos. It identifies a replaceable planar sign, tracks that plane through camera motion, perspective-warps each branch asset, and exports final MP4 variants.

This repository contains a hackathon spike. The current tracker validates planar tracking and three-variant rendering. The next stage adds one vision-model call to locate the sign in the first frame, then reuses geometric tracking for the rest of the video.

## Current spike

```powershell
python spikes/franchiseflow/tracker.py `
  --video spikes/franchiseflow/fixtures/real_storefront.mp4 `
  --quad spikes/franchiseflow/quad.json `
  --signs spikes/franchiseflow/assets/riverside.png spikes/franchiseflow/assets/downtown.png spikes/franchiseflow/assets/northpark.png `
  --out-dir spikes/franchiseflow/outputs2
```

The current spike requires a first-frame quadrilateral and uses OpenCV SIFT, RANSAC homography, and LK optical-flow fallback. It reports failed frames instead of freezing a stale replacement over the video.

## Vision model configuration

Copy `.env.example` to `.env`, set `BANANA_API_KEY`, and run the auto-selection probe after the vision client lands. The key is read only from the environment. The client defaults to the GLM-5.3-flash model and accepts either configured gateway URL.

```powershell
python spikes/franchiseflow/auto_quad.py `
  --image spikes/franchiseflow/fixtures/real_storefront_first.jpg `
  --out-json spikes/franchiseflow/quad_auto.json `
  --preview spikes/franchiseflow/quad_auto_preview.jpg
```

The probe makes one OpenAI-compatible `/chat/completions` request and expects JSON corners. A gateway error stops the run with a visible error; it never falls back to a guessed rectangle. The local tracker can then consume the returned `quad`.

Run parser tests with:

```powershell
python -m unittest discover -s spikes/franchiseflow/tests -v
```

## Evidence

See [the 30-minute experiment report](docs/FranchiseFlow_30分钟技术实验.md) and the source/license record for the Pexels fixture in `spikes/franchiseflow/fixtures/source.md`.

## Credits

The fixture is from Pexels. OpenCV, NumPy, and Pillow are used in the tracking spike. The final submission will list the vision model and any API gateway used during the hackathon.
