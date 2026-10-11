# FranchiseFlow Productization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前三分支招牌替换 Demo 推进为支持自动候选验证、人工确认、1 到 10 个分支、音频保留和可查询任务状态的本地产品原型。

**Architecture:** 视觉客户端一次返回候选平面，候选验证器用首帧、中间帧和末帧检查几何与纹理条件。通过验证的候选交给现有 SIFT/RANSAC/LK 跟踪器，人工确认的候选保留原始候选和修正后的四角。任务 API 使用本地目录 JSON 加线程池保存状态，输出通过 FFmpeg 编码成浏览器可播放的 H.264 MP4。

**Tech Stack:** Python 3.11+, Flask, OpenCV SIFT/RANSAC/PyrLK, NumPy, Pillow, FFmpeg/libx264, 原生 HTML/CSS/JavaScript, unittest。

**Spec:** `docs/superpowers/specs/2026-10-12-franchiseflow-product-design.md`

## Global Constraints

- 第一版只承诺单条连续镜头、一个主要平面招牌、720p、15 秒以内、1 到 10 个分店素材。
- 模型坐标必须先经过 JSON、有限数值、凸四边形、面积和边界校验；失败时不得猜测矩形。
- `initialization_mode` 只允许 `vision_model`、`manual_fallback`、`reviewed_candidate`。
- 跟踪失败帧保留原画，不能沿用上一帧的替换结果。
- 输出必须为 H.264、`yuv420p`、`faststart` MP4；输入有音频时保留音频轨。
- API 密钥只能来自环境变量或被 Git 忽略的 `.env`，日志、JSON、截图和提交历史不得写入密钥。
- 不在本阶段加入 SAM、视频抠像、多场景切分、3D 网格、数据库或云端队列。

## Review Focus

- 模型返回混合归一化和像素坐标时，任务必须拒绝候选并进入 `vision_failed`；测试归属于任务 1。
- 候选在首帧合法但中间帧越界或纹理点不足时，任务必须进入 `needs_review`；测试归属于任务 1。
- `--quad-json` 使用人工文件时，摘要必须记录 `manual_fallback`，不能写成视觉模型定位；测试归属于任务 2。
- 输入没有音频和输入带音频都必须成功输出，且带音频的输出要有音频轨；测试归属于任务 2。
- 用户在任务运行期间刷新页面时，能够通过 `GET /api/jobs/<id>` 恢复状态和输出；测试归属于任务 3。

### Task 1: 候选定位与三帧验证

**Files:**
- Modify: `spikes/franchiseflow/vision_client.py:98-170`
- Create: `spikes/franchiseflow/candidate_validator.py`
- Create: `spikes/franchiseflow/video_samples.py`
- Test: `spikes/franchiseflow/tests/test_vision_client.py`
- Test: `spikes/franchiseflow/tests/test_candidate_validator.py`
- Test: `spikes/franchiseflow/tests/test_video_samples.py`

**Interfaces:**
- `VisionClient.locate_candidates(image_bytes: bytes, width: int, height: int, target_hint: str | None = None) -> dict[str, Any]` returns `{"model": str, "candidates": [{"quad": list[list[float]], "confidence": float, "reason": str}]}` with pixel-coordinate quads.
- `VisionClient.locate_sign(...) -> dict[str, Any]` remains as a compatibility wrapper that returns the first candidate or raises the existing readable error.
- `sample_video_frames(video: str | Path, count: int = 3) -> list[tuple[int, np.ndarray]]` returns first, middle and last decodable frames with their frame indices.
- `validate_candidate(samples: list[tuple[int, np.ndarray]], candidate: dict[str, Any], min_texture_points: int = 8) -> dict[str, Any]` returns `accepted`, `requires_human_review`, `reasons`, and `sample_metrics`.

- [ ] **Step 1: Write failing parsing and candidate tests**

  Add tests for a multi-candidate JSON response, normalized coordinates, missing candidates, mixed coordinate systems, and a candidate whose quad is convex in the first frame but out of bounds in a later sample. Assert that invalid candidates are rejected without fabricating a fallback quad.

- [ ] **Step 2: Run the focused tests and verify they fail**

  Run: `python -m unittest spikes.franchiseflow.tests.test_vision_client spikes.franchiseflow.tests.test_candidate_validator -v`

  Expected: FAIL because `locate_candidates`, `candidate_validator.py`, and the sample-based response shape do not exist yet.

- [ ] **Step 3: Implement `locate_candidates` and compatibility behavior**

  Update the prompt to request a list of zero or more planar sign candidates from the supplied contact-sheet image. Normalize every candidate with the existing `normalize_quad`, preserve `confidence` and `reason`, attach the actual configured model name, and raise a redacted `RuntimeError` for malformed or empty responses. Keep `locate_sign` delegating to the first candidate so the old CLI and tests remain usable.

- [ ] **Step 4: Implement `sample_video_frames` and `validate_candidate`**

  Read video metadata, choose indices at 0, midpoint, and last available frame, and return decoded frames. For each candidate, check image bounds, convexity, area ratio, SIFT keypoint count inside the polygon, and projected quad area/bounds against the other samples. Return machine-readable reasons such as `quad_out_of_bounds`, `insufficient_texture`, or `inconsistent_samples`; set `requires_human_review` whenever the candidate is structurally valid but not safe to auto-run.

- [ ] **Step 5: Run focused tests and commit**

  Run: `python -m unittest spikes.franchiseflow.tests.test_vision_client spikes.franchiseflow.tests.test_candidate_validator spikes.franchiseflow.tests.test_video_samples -v`

  Expected: PASS. Commit with `git add spikes/franchiseflow/vision_client.py spikes/franchiseflow/candidate_validator.py spikes/franchiseflow/video_samples.py spikes/franchiseflow/tests && git commit -m "feat: validate multi-frame sign candidates"`.

### Task 2: Generalized tracking, provenance, and audio-preserving encoding

**Files:**
- Modify: `spikes/franchiseflow/tracker.py:22-41,203-346`
- Modify: `spikes/franchiseflow/metrics.py`
- Modify: `spikes/franchiseflow/tests/test_tracker.py`
- Create: `spikes/franchiseflow/tests/test_audio_encoding.py`

**Interfaces:**
- `transcode_to_h264(input_path: Path, output_path: Path, audio_source: Path | None = None) -> Path` maps the raw video stream to `libx264`, `yuv420p`, `+faststart`, and optionally maps `audio_source` audio with AAC.
- `run_tracker(video, quad, signs, out_dir, illumination=False, max_frames=0, min_inliers=8, ransac_px=3.0, ratio=0.75, no_debug=False, initialization_mode="manual_fallback", preserve_audio=True, model_name=None, source_candidate=None) -> dict[str, Any]` accepts 1 to 10 sign paths and records provenance.
- `summary.json` includes `branch_count`, `initialization_mode`, `model`, `audio_policy`, `audio_present`, `frames`, `failed_frames`, `output_videos`, and the existing residual metrics.

- [ ] **Step 1: Write failing tracker and encoding tests**

  Extend the synthetic fixture to run with one and five signs. Add a test that passes `initialization_mode="manual_fallback"` and asserts the summary mode stays manual. Add subprocess command assertions for `libx264`, `yuv420p`, `+faststart`, optional audio mapping, and a test that missing FFmpeg raises before video processing.

- [ ] **Step 2: Run the focused tracker tests and verify the new cases fail**

  Run: `python -m unittest spikes.franchiseflow.tests.test_tracker spikes.franchiseflow.tests.test_audio_encoding -v`

  Expected: FAIL because the current implementation requires exactly three signs and always drops audio.

- [ ] **Step 3: Generalize `run_tracker` to 1 through 10 branches**

  Replace the exact-three check with the documented range, generate output names from every supplied branch, and keep the frame count and dimensions identical across all outputs. Reject zero or more than ten branches with a readable `ValueError`.

- [ ] **Step 4: Fix initialization provenance and add audio-aware transcoding**

  Remove the CLI `vision_quad` label. Pass `manual_fallback` for both `--quad` and `--quad-json` unless an explicit reviewed or model mode is supplied. In `transcode_to_h264`, use the raw output as video input and the source video as optional audio input, then use `ffprobe` or an equivalent check to record whether an audio stream was present. Preserve the no-audio path without adding a silent track.

- [ ] **Step 5: Run all tracker tests and verify real media metadata**

  Run: `python -m unittest spikes.franchiseflow.tests.test_tracker spikes.franchiseflow.tests.test_audio_encoding -v`.

  Then run `ffprobe -v error -show_streams -of json <output.mp4>` on a generated output and verify H.264/yuv420p, faststart, matching frame dimensions, and an audio stream when the fixture has one. Commit with `git add spikes/franchiseflow/tracker.py spikes/franchiseflow/metrics.py spikes/franchiseflow/tests && git commit -m "feat: support arbitrary branches and preserve audio"`.

### Task 3: Persistent local job orchestration and human review API

**Files:**
- Modify: `prototype/franchiseflow/app.py:1-210`
- Create: `prototype/franchiseflow/job_store.py`
- Modify: `prototype/franchiseflow/tests/test_app.py`
- Create: `prototype/franchiseflow/tests/test_job_store.py`

**Interfaces:**
- `POST /api/jobs` accepts one MP4 and 1 to 10 `branches` files plus optional `target_hint`, `use_manual_quad`, and `manual_quad`; returns HTTP 202 with `job_id` and `status="queued"`.
- `GET /api/jobs/<job_id>` returns `queued`, `running`, `needs_review`, `completed`, or `failed`, with phase, error, candidate preview metadata, and outputs when available.
- `POST /api/jobs/<job_id>/review` accepts `{"quad": [[x,y], ...]}` and resumes a `needs_review` task with `initialization_mode="reviewed_candidate"`.
- `GET /api/jobs/<job_id>/outputs` returns the completed branch URLs and summary.
- `POST /api/render` remains a synchronous compatibility wrapper that creates a job and waits for its terminal state.

- [ ] **Step 1: Write failing API and persistence tests**

  Test 1, 3, and 10 branch uploads, malformed uploads, `202 queued`, status polling, a mocked `needs_review` result, review submission, completed output retrieval, and restart-safe JSON state. Assert that review rejects unknown job IDs, non-reviewable states, invalid quads, and mismatched job ownership.

- [ ] **Step 2: Run focused app tests and verify they fail**

  Run: `python -m unittest prototype.franchiseflow.tests.test_app prototype.franchiseflow.tests.test_job_store -v`

  Expected: FAIL because `/api/jobs` POST, review, outputs, and persisted state are not implemented.

- [ ] **Step 3: Implement `job_store.py`**

  Add atomic JSON writes under `prototype/franchiseflow/outputs/<job_id>/job.json`, safe filename/path helpers, allowed state transitions, and `load_job`/`save_job` functions. Store only filenames, normalized quads, model names, metrics, and errors; never store API keys or base64 images.

- [ ] **Step 4: Implement the worker and job endpoints in `app.py`**

  Use a bounded `ThreadPoolExecutor(max_workers=1)` for local processing. The worker saves uploads, samples frames, calls `VisionClient.locate_candidates`, validates candidates, writes previews, and either enters `needs_review` or calls `run_tracker`. The review endpoint validates the submitted quad and schedules the same worker with `reviewed_candidate`. Keep `JOBS` as a compatibility cache populated from `job_store`.

- [ ] **Step 5: Run focused tests and commit**

  Run: `python -m unittest prototype.franchiseflow.tests.test_app prototype.franchiseflow.tests.test_job_store -v`.

  Expected: PASS, including a test that polls after the worker has completed and receives the same state from disk. Commit with `git add prototype/franchiseflow/app.py prototype/franchiseflow/job_store.py prototype/franchiseflow/tests && git commit -m "feat: add persistent local render jobs"`.

### Task 4: Multi-branch and review-aware web interface

**Files:**
- Modify: `prototype/franchiseflow/templates/index.html`
- Modify: `prototype/franchiseflow/static/app.js`
- Modify: `prototype/franchiseflow/static/app.css`
- Modify: `prototype/franchiseflow/tests/test_app.py`

**Interfaces:**
- The form can add or remove branch image inputs up to ten and sends them as repeated `branches` fields.
- The client polls `GET /api/jobs/<job_id>` every 750 ms until `completed`, `needs_review`, or `failed`.
- A `needs_review` result displays the candidate preview and draggable four corners, then posts the corrected quad to `/api/jobs/<job_id>/review`.
- The output grid is generated from the returned branch list, rather than assuming exactly three players.

- [ ] **Step 1: Add browser-contract tests**

  Extend the Flask page test to assert the page contains the job polling endpoint, review controls, and no copy claiming perfect or zero-jitter output. Keep API tests for output counts in Task 3.

- [ ] **Step 2: Implement dynamic branch controls and job polling**

  Replace the three fixed file inputs with a branch list initialized to three rows, add/remove controls, and client-side validation for 1 to 10 images. Submit to `/api/jobs`, render phase and model provenance, and stop polling on terminal states.

- [ ] **Step 3: Implement manual review interaction**

  Draw the candidate quad on the sampled frame, allow pointer dragging of four handles, send pixel coordinates after bounds validation, and show the reasons returned by the validator. The UI must label a reviewed candidate as human-reviewed.

- [ ] **Step 4: Render arbitrary output cards and validate playback**

  Build one `<video>` and download link per returned branch, attach `canplay`/`error` handlers, and display a useful browser error if the server cannot satisfy a Range request. Preserve the prebuilt demo loader as a separate convenience path.

- [ ] **Step 5: Run browser-contract tests and manually smoke test**

  Run: `python -m unittest prototype.franchiseflow.tests.test_app -v`.

  Start with `python prototype/franchiseflow/app.py`, upload a fixture with 1, 3, and 5 branches, exercise the review path, drag the video timeline, and confirm the output grid and audio behavior. Commit with `git add prototype/franchiseflow/templates/index.html prototype/franchiseflow/static/app.js prototype/franchiseflow/static/app.css prototype/franchiseflow/tests && git commit -m "feat: add multi-branch reviewable web flow"`.

### Task 5: Reproducibility, packaging, and release evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/FranchiseFlow_Demo脚本.md`
- Modify: `docs/FranchiseFlow_发布前修正与实测报告.md`
- Create: `requirements.txt`
- Create: `LICENSE`
- Modify: `.gitignore`
- Create: `spikes/franchiseflow/tests/test_release_contract.py`

**Interfaces:**
- README documents the first-version contract, automatic and reviewed paths, exact setup command, FFmpeg requirement, API endpoints, audio behavior, and measured limits.
- Demo script presents the product flow with honest model provenance and keeps the recording under 180 seconds.
- A clean clone can install dependencies, generate a short fixture, start the app, and run the full test suite without ignored output directories.

- [ ] **Step 1: Write release-contract tests**

  Assert that README and demo copy contain no `Zero Jitter`, `100%`, `广播级`, or claims that residual error equals physical boundary error. Assert that `requirements.txt`, `LICENSE`, and the fixture-generation command exist, and that tracked files contain no secret patterns.

- [ ] **Step 2: Run the release-contract tests and verify stale copy is found**

  Run: `python -m unittest spikes.franchiseflow.tests.test_release_contract -v`.

  Expected: FAIL until the stale claims, missing packaging files, and clean-clone instructions are corrected.

- [ ] **Step 3: Update packaging and documentation**

  Add pinned minimum dependencies used by the code, an MIT license, a clean-clone setup path, exact sample-generation command, endpoint examples, measured 6-second and 20-frame evidence boundaries, the absence of occlusion matting, and the actual model default. Rewrite the demo script to show input, candidate/review state, outputs, and technical explanation within 2:45.

- [ ] **Step 4: Run the full release gate**

  Run:

  `python -m unittest discover -s spikes/franchiseflow/tests -v`

  `python -m unittest discover -s prototype/franchiseflow/tests -v`

  `git diff --check`

  `git grep -n -I -E 'sk-[A-Za-z0-9]{20,}|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}' -- ':!D*'`

  Then use `ffprobe` on one no-audio and one audio-bearing output, and run a clean-clone smoke test in a temporary directory. Expected: all tests pass, the secret scan returns no matches, and both media paths are browser-playable.

- [ ] **Step 5: Commit and publish release evidence**

  Commit with `git add README.md docs/FranchiseFlow_Demo脚本.md docs/FranchiseFlow_发布前修正与实测报告.md requirements.txt LICENSE .gitignore spikes/franchiseflow/tests/test_release_contract.py && git commit -m "docs: publish productization release contract"`, push `master`, and record the exact commit plus test commands in the release report.

