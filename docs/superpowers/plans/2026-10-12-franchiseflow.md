# FranchiseFlow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the verified planar-sign tracking spike into a demo-ready tool that uses one GLM-5.3-flash vision call to locate a storefront sign, reuses local geometric tracking for the video, and exports three branch-specific MP4 files.

**Architecture:** The first frame is sent once to an OpenAI-compatible vision endpoint. The model returns four pixel corners for the replaceable planar sign. OpenCV then tracks that plane with first-frame SIFT/RANSAC and LK fallback, composites each branch asset with a perspective transform, and writes three videos. A small local Flask page exposes upload, render, status, and video preview; the CLI remains the reproducible fallback.

**Tech Stack:** Python 3.11+, OpenCV 5.x, NumPy, Pillow, Flask, vanilla HTML/CSS/JS, OpenAI-compatible HTTP API.

**Spec:** `docs/FranchiseFlow_30分钟技术实验.md`

**Execution owner:** Antigravity. Start from the public repository's current `master` branch. Commits `184ed16`, `25c7f2e`, and `971295f` already contain the tracking spike, the vision client, its focused test, and the gateway access record. Continue incrementally; do not recreate those files from scratch.

## Global Constraints

- Read `BANANA_API_KEY`, `BANANA_API_BASE_URL`, and `BANANA_VISION_MODEL` from the environment or an ignored `.env` file only.
- Never place a secret in source, command-line arguments, JSON output, logs, screenshots, Git history, or the public repository.
- The default model is `GLM-5.3-flash`; the default base URL is `https://e.c.h-e.top/v1`, and both are configurable.
- One video job makes one vision request, then uses local tracking for all later frames.
- A failed API request or invalid quad must stop the job with a visible error. Do not guess a rectangle or claim success.
- A failed frame must keep the original frame and be counted in the tracking report. Do not freeze a stale replacement over a failed frame.
- Generated MP4 files, temporary uploads, caches, and `.env` stay ignored by Git.
- The current Pexels fixture and its license record remain available for reproducible testing.
- The demo should show a real final video artifact, not a report or a quality score.

## Review Focus

- **Gateway access:** HTTP 401/403, timeout, malformed response, or model refusal must give a useful error and leave no partial success output. Test in Task 1.
- **Model coordinates:** pixel coordinates, normalized coordinates, fenced JSON, missing quad, concave/tiny/out-of-frame quad. Test in Task 1.
- **Camera motion:** planar perspective change and brief feature loss must recover or be counted as failure. Test in Task 2.
- **Occlusion:** a foreground object crossing the sign must not produce a floating or frozen branch overlay. Test in Task 2 and choose a no-occlusion demo cut if the fixture cannot pass.
- **Three-branch consistency:** every branch must have the same frame count, dimensions, and tracked corner sequence. Test in Task 3.
- **Secret hygiene:** repository scan must find no API key or bearer token before every push. Test in Task 4.

---

### Task 1: Lock the vision API contract and smoke test

**Files:**
- Modify: `spikes/franchiseflow/vision_client.py`
- Modify: `spikes/franchiseflow/auto_quad.py`
- Test: `spikes/franchiseflow/tests/test_vision_client.py`
- Modify: `docs/FranchiseFlow_API接入记录.md`

**Interfaces:**
- Consumes: `VisionClient(api_key, base_url, model, timeout)` and an image byte string.
- Produces: `VisionClient.locate_sign(image_bytes, width, height) -> dict` with `quad`, `confidence`, `reason`, and `model`.
- `quad` is four pixel corners in TL, TR, BR, BL order.

- [ ] **Step 1: Add failing tests for the HTTP response contract.**

  Add tests for an OpenAI-compatible `choices[0].message.content` response, an HTTP 403, a timeout, and a response with `quad: null`. Assert that only the expected exception type and redacted error message are returned.

- [ ] **Step 2: Run the focused tests and confirm the new failure cases fail.**

  Run `python -m unittest discover -s spikes/franchiseflow/tests -v`.

- [ ] **Step 3: Implement the smallest contract-preserving client.**

  Keep the current environment-only secret handling. Do not add a third-party SDK. Normalize unit coordinates only when every coordinate is in `[0, 1]`; reject mixed units. Keep response-body text out of raised errors.

- [ ] **Step 4: Run the focused tests and the live smoke test.**

  Run the unit command above. Then run:

  ```powershell
  $env:BANANA_API_KEY = "<set only in the current process>"
  python spikes/franchiseflow/auto_quad.py `
    --image spikes/franchiseflow/fixtures/real_storefront_first.jpg `
    --out-json spikes/franchiseflow/quad_auto.json `
    --preview spikes/franchiseflow/quad_auto_preview.jpg
  ```

  Expected success is a valid four-corner JSON and preview. Expected failure for the current gateway is a clear HTTP 403 message with no output JSON and no secret in logs. Record the actual result in `docs/FranchiseFlow_API接入记录.md`.

- [ ] **Step 5: Commit.**

  ```powershell
  git add spikes/franchiseflow docs/FranchiseFlow_API接入记录.md
  git commit -m "test: lock vision gateway contract"
  git push origin master
  ```

### Task 2: Make tracking and compositing demo-safe

**Files:**
- Modify: `spikes/franchiseflow/tracker.py`
- Create: `spikes/franchiseflow/metrics.py`
- Test: `spikes/franchiseflow/tests/test_tracker.py`
- Create: `spikes/franchiseflow/fixtures/demo_cut.mp4` only if a trimmed fixture is required; keep it ignored if generated locally.

**Interfaces:**
- Consumes: a video path, a validated quad, and exactly three image paths.
- Produces: three MP4s, `tracking.jsonl`, and `summary.json` with frame count, failed frames, median inliers, and reprojection percentiles.
- `run_tracker(video, quad, signs, out_dir, illumination=False) -> dict` should expose the same summary fields as the CLI.

- [ ] **Step 1: Write synthetic motion tests.**

  Create a small textured planar fixture with known homographies. Assert that the returned quad error is under 3 px, all three outputs have the same frame count, and a forced feature-loss frame is counted as failed rather than painted with a stale quad.

- [ ] **Step 2: Run the tests and confirm the missing callable interface fails.**

  Run `python -m unittest spikes/franchiseflow/tests/test_tracker.py -v`.

- [ ] **Step 3: Extract the current CLI body into `run_tracker(...)`.**

  Keep the current SIFT anchor matching, RANSAC validation, LK forward-backward fallback, and explicit failed-frame policy. Add an optional `--quad-json` input from Task 1. Preserve the manual `--quad` flag as an engineering fallback and label it in `summary.json`.

- [ ] **Step 4: Validate the real fixture.**

  Run the full 9.7-second fixture. If the 7-frame foreground obstruction remains, use a clearly labeled no-obstruction cut for the polished Demo and keep the full fixture as a stress test. Do not remove failed frames from the metrics.

- [ ] **Step 5: Commit.**

  ```powershell
  git add spikes/franchiseflow
  git commit -m "feat: expose safe planar tracking pipeline"
  git push origin master
  ```

### Task 3: Add the smallest usable local Demo page

**Files:**
- Create: `prototype/franchiseflow/app.py`
- Create: `prototype/franchiseflow/templates/index.html`
- Create: `prototype/franchiseflow/static/app.js`
- Create: `prototype/franchiseflow/static/app.css`
- Test: `prototype/franchiseflow/tests/test_app.py`

**Interfaces:**
- `POST /api/render` accepts one MP4 and three PNG/JPG branch assets.
- `GET /api/jobs/<job_id>` returns `status`, `vision_quad`, `summary`, and output URLs.
- `GET /outputs/<filename>` serves only files inside the ignored job output directory.

- [ ] **Step 1: Write API tests for upload validation and job status.**

  Assert that missing video, fewer than three branch assets, unsupported extensions, and an API failure return 4xx/5xx JSON without leaving a fake result. Assert that a successful mocked run returns three output URLs and the tracking summary.

- [ ] **Step 2: Implement a synchronous single-job page.**

  Keep the page to one upload form, one Generate button, a four-corner preview, a progress/status line, and three video cards. Use a deterministic per-job directory under the ignored output path. Do not add accounts, databases, dashboards, or batch history.

- [ ] **Step 3: Wire the page to the pipeline.**

  Extract the first frame, call `VisionClient` once, pass the returned quad to `run_tracker`, and display the actual summary. The UI must show “manual initialization” only when the explicit engineering fallback is used.

- [ ] **Step 4: Run an end-to-end local test with the API client mocked.**

  Run `python -m unittest discover -s prototype/franchiseflow/tests -v`, then start `python prototype/franchiseflow/app.py` and verify a browser can load the page and play all three outputs.

- [ ] **Step 5: Commit.**

  ```powershell
  git add prototype/franchiseflow
  git commit -m "feat: add FranchiseFlow demo page"
  git push origin master
  ```

### Task 4: Prepare the submission artifact and final verification

**Files:**
- Create: `docs/FranchiseFlow_Demo脚本.md`
- Modify: `README.md`
- Modify: `.gitignore` if any new generated path is introduced

- [ ] **Step 1: Write the three-minute recording script.**

  Use this sequence:

  - 0:00–0:20, show one moving storefront clip and the repeated work of making three branch versions.
  - 0:20–0:45, upload the clip and three branch assets.
  - 0:45–1:10, show the vision model locating the physical sign plane.
  - 1:10–1:50, show three final MP4s playing side by side as the camera moves.
  - 1:50–2:25, show the pipeline split between one vision call and local SIFT/RANSAC/LK tracking.
  - 2:25–2:45, show metrics and the known obstruction limitation.
  - 2:45–3:00, show GitHub, credits, and the final artifact.

- [ ] **Step 2: Add the submission checklist.**

  Include project name, team, public GitHub URL, public YouTube video no longer than three minutes, brainstorming/inspiration/design process/resources/final product, API and model credits, and the exact run command. Banana Hacks requires a public GitHub link and a strict three-minute public YouTube video; the project must be created in the hacking window. Sources: `https://banana-hacks-1111.devpost.com/` and `https://www.bananahacks.tech/code-of-conduct`.

- [ ] **Step 3: Run the final checks.**

  ```powershell
  python -m unittest discover -s spikes/franchiseflow/tests -v
  python -m unittest discover -s prototype/franchiseflow/tests -v
  git diff --check
  git grep -n -I -E 'sk-[A-Za-z0-9]{20,}|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}' -- ':!D*'
  git status --short
  ```

  The secret scan must produce no matches. Generated output directories must remain ignored. The recorded Demo must play from start to finish without relying on a live API request.

- [ ] **Step 4: Commit and push the submission docs.**

  ```powershell
  git add README.md docs/FranchiseFlow_Demo脚本.md .gitignore
  git commit -m "docs: prepare FranchiseFlow submission"
  git push origin master
  ```

## Stop Conditions

- If the GLM gateway remains HTTP 403 and no approved alternate endpoint is available, stop API-dependent work and report the blocker. Do not hide it with a guessed quad.
- If automatic localization works but tracking fails on the curated demo clip, use the shorter no-obstruction clip and keep the full clip as a stress test. Do not claim general occlusion handling.
- If rendering three variants takes longer than the recording window, pre-render the exact demo inputs and show the reproducible command in the README. The video must still show the actual generated artifacts and the code path.
- If a task requires a new product subsystem outside the four tasks above, stop and ask for a new plan instead of expanding scope.
