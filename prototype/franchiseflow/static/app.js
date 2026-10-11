document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('render-form');
  const videoInput = document.getElementById('video-input');
  const videoFileName = document.getElementById('video-file-name');
  const branchesList = document.getElementById('branches-list');
  const addBranchButton = document.getElementById('btn-add-branch');
  const btnSubmit = document.getElementById('btn-submit');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnText = btnSubmit.querySelector('.btn-text');
  const btnLoadSample = document.getElementById('btn-load-sample');
  const quadStatusBadge = document.getElementById('quad-status-badge');
  const placeholderPreview = document.getElementById('placeholder-preview');
  const previewCanvas = document.getElementById('preview-canvas');
  const reviewPanel = document.getElementById('review-panel');
  const reviewReasons = document.getElementById('review-reasons');
  const reviewButton = document.getElementById('review-quad');
  const outputsGrid = document.getElementById('outputs-grid');
  const metricFrames = document.getElementById('metric-frames');
  const metricTime = document.getElementById('metric-time');
  const metricInliers = document.getElementById('metric-inliers');
  const metricReproj = document.getElementById('metric-reproj');
  const metricFailed = document.getElementById('metric-failed');
  const metricInitMode = document.getElementById('metric-init-mode');
  const metricScenes = document.getElementById('metric-scenes');
  const metricSurfaces = document.getElementById('metric-surfaces');
  const sceneReviewControls = document.getElementById('scene-review-controls');
  const scenePicker = document.getElementById('review-scene-picker');
  const surfacePicker = document.getElementById('review-surface-picker');
  const sceneReviewSummary = document.getElementById('scene-review-summary');

  let branchInputs = [];
  let activeJobId = null;
  let reviewQuad = null;
  let dragIndex = -1;
  let reviewBackground = null;
  let reviewScenes = [];
  let activeSceneId = null;
  let activeSurfaceIndex = 0;
  let reviewPreviewUrl = null;
  let reviewBranchCount = 1;
  const sceneQuads = new Map();
  const sceneSurfaces = new Map();
  let previewRenderToken = 0;

  function addBranchInput() {
    if (branchInputs.length >= 10) return;
    const index = branchInputs.length + 1;
    const row = document.createElement('div');
    row.className = 'branch-upload-box';
    row.dataset.index = String(index);
    row.innerHTML = `
      <label class="branch-label" for="branch-input-${index}">分支 ${index}</label>
      <input type="file" id="branch-input-${index}" name="branches" accept="image/png,image/jpeg">
      <div class="branch-file-name">选择图片 ${index}</div>
      <button type="button" class="branch-remove btn btn-outline">移除</button>`;
    const input = row.querySelector('input');
    const name = row.querySelector('.branch-file-name');
    input.addEventListener('change', () => {
      name.textContent = input.files[0] ? input.files[0].name : `选择图片 ${index}`;
    });
    row.querySelector('.branch-remove').addEventListener('click', () => {
      row.remove();
      branchInputs = branchInputs.filter(item => item !== input);
      renumberBranches();
    });
    branchesList.appendChild(row);
    branchInputs.push(input);
    updateAddButton();
  }

  function renumberBranches() {
    [...branchesList.children].forEach((row, i) => {
      const index = i + 1;
      row.dataset.index = String(index);
      row.querySelector('.branch-label').textContent = `分支 ${index}`;
      row.querySelector('input').id = `branch-input-${index}`;
      row.querySelector('.branch-label').htmlFor = `branch-input-${index}`;
    });
    updateAddButton();
  }

  function updateAddButton() {
    addBranchButton.disabled = branchInputs.length >= 10;
    addBranchButton.textContent = branchInputs.length >= 10 ? '已达到 10 个分支上限' : '添加分支';
  }

  for (let i = 0; i < 3; i++) addBranchInput();
  addBranchButton.addEventListener('click', addBranchInput);

  videoInput.addEventListener('change', () => {
    if (videoInput.files[0]) {
      videoFileName.textContent = videoInput.files[0].name;
      drawVideoFirstFrame(videoInput.files[0], reviewQuad);
    }
  });

  function setStatus(text, className = 'badge badge-tech') {
    quadStatusBadge.textContent = text;
    quadStatusBadge.className = className;
  }

  function drawQuadOverlay(ctx, quad, editable = false) {
    if (!quad || quad.length !== 4) return;
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#06b6d4';
    ctx.fillStyle = 'rgba(6, 182, 212, 0.2)';
    ctx.beginPath();
    ctx.moveTo(quad[0][0], quad[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(quad[i][0], quad[i][1]);
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
    quad.forEach((point, index) => {
      ctx.fillStyle = editable ? '#f59e0b' : '#ffffff';
      ctx.beginPath();
      ctx.arc(point[0], point[1], editable ? 9 : 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.fillText(['TL', 'TR', 'BR', 'BL'][index], point[0] + 10, point[1] - 5);
    });
  }

  function drawSurfaceOverlays(ctx, surfaces, activeIndex = -1) {
    (surfaces || []).forEach((surface, index) => {
      if (surface && surface.quad) drawQuadOverlay(ctx, surface.quad, index === activeIndex);
    });
  }

  function drawVideoFirstFrame(file, quad = null, frameIndex = 0, surfaces = null) {
    const renderToken = ++previewRenderToken;
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.playsInline = true;
    video.src = URL.createObjectURL(file);
    video.onloadeddata = () => {
      const fps = Number(video.dataset.fps || 30);
      video.currentTime = Math.max(0, Number(frameIndex) || 0) / fps;
    };
    video.onseeked = () => {
      if (renderToken !== previewRenderToken) return;
      placeholderPreview.style.display = 'none';
      previewCanvas.style.display = 'block';
      previewCanvas.width = video.videoWidth || 640;
      previewCanvas.height = video.videoHeight || 360;
      const resolvedQuad = quad || (!reviewPanel.hidden ? centeredQuad() : null);
      if (!quad && resolvedQuad && !reviewPanel.hidden) {
        reviewQuad = cloneQuad(resolvedQuad);
        if (activeSceneId != null) sceneQuads.set(String(activeSceneId), cloneQuad(resolvedQuad));
      }
      const ctx = previewCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
      reviewBackground = document.createElement('canvas');
      reviewBackground.width = previewCanvas.width;
      reviewBackground.height = previewCanvas.height;
      reviewBackground.getContext('2d').drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
      if (surfaces && surfaces.length) drawSurfaceOverlays(ctx, surfaces, activeSurfaceIndex);
      else drawQuadOverlay(ctx, resolvedQuad, Boolean(reviewPanel && !reviewPanel.hidden));
      URL.revokeObjectURL(video.src);
    };
  }

  function drawSampleFrame(url, quad, surfaces = null) {
    const renderToken = ++previewRenderToken;
    const image = new Image();
    image.onload = () => {
      if (renderToken !== previewRenderToken) return;
      placeholderPreview.style.display = 'none';
      previewCanvas.style.display = 'block';
      previewCanvas.width = image.width;
      previewCanvas.height = image.height;
      const resolvedQuad = quad || (!reviewPanel.hidden ? centeredQuad() : null);
      if (!quad && resolvedQuad && !reviewPanel.hidden) {
        reviewQuad = cloneQuad(resolvedQuad);
        if (activeSceneId != null) sceneQuads.set(String(activeSceneId), cloneQuad(resolvedQuad));
      }
      const ctx = previewCanvas.getContext('2d');
      ctx.drawImage(image, 0, 0);
      reviewBackground = document.createElement('canvas');
      reviewBackground.width = image.width;
      reviewBackground.height = image.height;
      reviewBackground.getContext('2d').drawImage(image, 0, 0);
      if (surfaces && surfaces.length) drawSurfaceOverlays(ctx, surfaces, activeSurfaceIndex);
      else drawQuadOverlay(ctx, resolvedQuad, Boolean(reviewPanel && !reviewPanel.hidden));
    };
    image.src = url;
  }

  function cloneQuad(quad) {
    return Array.isArray(quad) && quad.length === 4
      ? quad.map(point => [Number(point[0]), Number(point[1])])
      : null;
  }

  function candidateQuad(scene) {
    const surface = scene && Array.isArray(scene.surfaces) ? scene.surfaces[0] : null;
    const candidates = [
      surface && (surface.quad || surface.pixel_quad),
      scene && scene.candidate && scene.candidate.quad,
      scene && scene.vision_candidate && scene.vision_candidate.quad,
      scene && scene.quad,
    ];
    for (const candidate of candidates) {
      const quad = cloneQuad(candidate);
      if (quad && quad.every(point => point.every(Number.isFinite))) return quad;
    }
    return null;
  }

  function candidateSurfaces(scene) {
    const source = scene && Array.isArray(scene.surfaces) && scene.surfaces.length
      ? scene.surfaces
      : (scene && Array.isArray(scene.candidates) ? scene.candidates : []);
    const surfaces = source.map((item, index) => ({
      surface_id: String(item.surface_id || `${sceneKey(scene)}_surface_${String(index + 1).padStart(3, '0')}`),
      quad: cloneQuad(item.quad || item.pixel_quad),
    })).filter(item => item.quad);
    if (surfaces.length) return surfaces;
    const fallback = candidateQuad(scene);
    return fallback ? [{ surface_id: `${sceneKey(scene)}_surface_001`, quad: fallback }] : [];
  }

  function centeredQuad() {
    const width = previewCanvas.width || 640;
    const height = previewCanvas.height || 360;
    return [
      [width * 0.20, height * 0.35],
      [width * 0.80, height * 0.35],
      [width * 0.80, height * 0.65],
      [width * 0.20, height * 0.65],
    ];
  }

  function sceneKey(scene) {
    return String(scene && scene.scene_id != null ? scene.scene_id : 'scene_001');
  }

  function sceneInitMode(scene) {
    const surface = scene && Array.isArray(scene.surfaces) ? scene.surfaces[0] : null;
    return (scene && scene.initialization_mode) || (surface && surface.initialization_mode) || '待确认';
  }

  function sceneRange(scene) {
    const start = scene && scene.start_frame != null ? scene.start_frame : '?';
    const end = scene && scene.end_frame != null ? scene.end_frame : '?';
    return `帧 ${start} 至 ${end}`;
  }

  function sceneStatus(scene) {
    const status = scene && scene.status ? String(scene.status) : 'needs_review';
    return ({ok: '已通过', accepted: '已通过', validated: '已通过', reviewed: '已复核', needs_review: '需确认', failed: '跟踪失败', queued: '待处理'})[status] || status;
  }

  function scenePreview(scene) {
    if (!scene) return null;
    return scene.preview || scene.sample_preview || scene.preview_url || null;
  }

  function updateSceneReviewMeta(scene) {
    if (!sceneReviewSummary) return;
    if (!scene) {
      sceneReviewSummary.textContent = '';
      return;
    }
    const surface = Array.isArray(scene.surfaces) ? scene.surfaces[0] : null;
    const surfaceStatus = surface && surface.status ? ` · 平面 ${sceneStatus(surface)}` : '';
    const surfaceCount = candidateSurfaces(scene).length;
    sceneReviewSummary.textContent = `${sceneRange(scene)} · 初始化 ${sceneInitMode(scene)} · ${sceneStatus(scene)} · ${surfaceCount || 0} 个平面${surfaceStatus}`;
    const reasons = (scene.validation && scene.validation.reasons) || (scene.candidate && scene.candidate.validation && scene.candidate.validation.reasons);
    reviewReasons.textContent = (reasons && reasons.length) ? reasons.join('、') : (sceneStatus(scene) === '需确认' ? '该场景的候选需要人工确认。' : '请检查招牌四角后提交。');
  }

  function drawReviewScene(scene) {
    const quad = activeSceneId == null ? reviewQuad : sceneQuads.get(String(activeSceneId));
    const surfaces = activeSceneId == null ? null : sceneSurfaces.get(String(activeSceneId));
    const preview = scenePreview(scene) || reviewPreviewUrl;
    if (preview) drawSampleFrame(preview, quad, surfaces);
    else if (videoInput.files[0]) drawVideoFirstFrame(videoInput.files[0], quad, scene && scene.start_frame, surfaces);
    else {
      previewRenderToken += 1;
      reviewBackground = null;
      redrawReview();
    }
    updateSceneReviewMeta(scene);
  }

  function saveActiveSceneQuad() {
    if (activeSceneId == null || !reviewQuad) return;
    const surfaces = sceneSurfaces.get(String(activeSceneId));
    if (surfaces && surfaces[activeSurfaceIndex]) {
      surfaces[activeSurfaceIndex].quad = cloneQuad(reviewQuad);
      if (activeSurfaceIndex === 0) sceneQuads.set(String(activeSceneId), cloneQuad(reviewQuad));
    } else {
      sceneQuads.set(String(activeSceneId), cloneQuad(reviewQuad));
    }
  }

  function switchReviewScene(sceneId) {
    saveActiveSceneQuad();
    const scene = reviewScenes.find(item => sceneKey(item) === String(sceneId));
    if (!scene) return;
    activeSceneId = sceneKey(scene);
    activeSurfaceIndex = 0;
    reviewQuad = cloneQuad(sceneQuads.get(activeSceneId));
    if (surfacePicker) {
      const surfaces = sceneSurfaces.get(activeSceneId) || [];
      surfacePicker.innerHTML = '';
      surfaces.forEach((surface, index) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `${surface.surface_id} · 平面 ${index + 1}`;
        surfacePicker.appendChild(option);
      });
      surfacePicker.value = '0';
    }
    if (scenePicker && scenePicker.value !== activeSceneId) scenePicker.value = activeSceneId;
    drawReviewScene(scene);
  }

  function setupSceneReview(job) {
    // The backend caps detection at eight hard-cut scenes. Keep every scene
    // in the review picker so a later shot cannot silently disappear from the
    // submitted scene plan.
    reviewScenes = Array.isArray(job.scenes) ? job.scenes.slice(0, 8) : [];
    sceneQuads.clear();
    sceneSurfaces.clear();
    activeSceneId = null;
    if (!reviewScenes.length) {
      if (sceneReviewControls) sceneReviewControls.hidden = true;
      return false;
    }
    reviewScenes.forEach(scene => {
      const surfaces = candidateSurfaces(scene);
      if (surfaces.length) {
        sceneSurfaces.set(sceneKey(scene), surfaces);
        sceneQuads.set(sceneKey(scene), surfaces[0].quad);
      }
    });
    if (sceneReviewControls) sceneReviewControls.hidden = false;
    if (scenePicker) {
      scenePicker.innerHTML = '';
      reviewScenes.forEach(scene => {
        const option = document.createElement('option');
        option.value = sceneKey(scene);
        option.textContent = `${sceneKey(scene)} · ${sceneRange(scene)} · ${sceneStatus(scene)}`;
        scenePicker.appendChild(option);
      });
      scenePicker.onchange = () => switchReviewScene(scenePicker.value);
    }
    switchReviewScene(sceneKey(reviewScenes[0]));
    return true;
  }

  function redrawReview() {
    if (!reviewBackground) return;
    const ctx = previewCanvas.getContext('2d');
    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    ctx.drawImage(reviewBackground, 0, 0, previewCanvas.width, previewCanvas.height);
    const surfaces = activeSceneId == null ? null : sceneSurfaces.get(String(activeSceneId));
    if (surfaces && surfaces.length) drawSurfaceOverlays(ctx, surfaces, activeSurfaceIndex);
    else drawQuadOverlay(ctx, reviewQuad, true);
  }

  function updateMetrics(summary, mode, job = null) {
    metricFrames.textContent = summary.frames ? `${summary.frames} 帧 (${summary.fps || 0} fps)` : '-';
    metricTime.textContent = summary.processing_seconds != null ? `${summary.processing_seconds}s` : '-';
    metricInliers.textContent = summary.median_inliers != null ? Math.round(summary.median_inliers) : '-';
    metricReproj.textContent = summary.reprojection_median_px != null ? `${summary.reprojection_median_px.toFixed(2)} px` : '-';
    metricFailed.textContent = summary.failed_frames ? `${summary.failed_frames.length} 帧` : '-';
    metricInitMode.textContent = mode || summary.initialization_mode || '-';
    if (metricScenes) metricScenes.textContent = summary.scene_count != null ? summary.scene_count : (job && Array.isArray(job.scenes) ? job.scenes.length : '-');
    const inferredSurfaces = job && Array.isArray(job.scenes)
      ? job.scenes.reduce((count, scene) => count + (Array.isArray(scene.surfaces) ? scene.surfaces.length : 0), 0)
      : null;
    if (metricSurfaces) metricSurfaces.textContent = summary.surface_count != null ? summary.surface_count : (inferredSurfaces || '-');
  }

  function renderOutputs(urls) {
    outputsGrid.innerHTML = '';
    (urls || []).forEach((url, index) => {
      const card = document.createElement('div');
      card.className = 'output-card';
      card.innerHTML = `<div class="output-card-header"><span class="branch-tag">Branch ${index + 1}</span><span class="branch-title">分支门店 ${index + 1}</span></div><div class="video-container"><video controls playsinline loop></video></div><div class="output-actions"><a class="btn btn-outline" download>下载成片 MP4</a></div>`;
      const player = card.querySelector('video');
      const link = card.querySelector('a');
      player.src = url;
      player.addEventListener('error', () => {
        card.querySelector('.branch-title').textContent += '（浏览器无法播放）';
      });
      link.href = url;
      link.download = `branch_${index + 1}.mp4`;
      outputsGrid.appendChild(card);
    });
  }

  function showReview(job) {
    reviewPanel.hidden = false;
    reviewBranchCount = Math.max(1, Number(job.branch_count) || 1);
    reviewPreviewUrl = job.sample_preview || job.preview || null;
    const multiScene = setupSceneReview(job);
    if (job.asset_bindings_required) {
      reviewReasons.textContent = '检测到多个可替换平面。当前提交会把每个分支素材明确应用到全部已复核平面；需要逐平面分配时请改用 API 的 asset_bindings。';
    }
    if (multiScene) return;
    if (sceneReviewControls) sceneReviewControls.hidden = true;
    activeSceneId = null;
    reviewQuad = cloneQuad(job.candidate && job.candidate.quad);
    reviewReasons.textContent = (job.validation && job.validation.reasons ? job.validation.reasons.join('、') : job.error || '候选没有通过三帧检查。');
    if (videoInput.files[0]) drawVideoFirstFrame(videoInput.files[0], reviewQuad);
    else if (reviewPreviewUrl) drawSampleFrame(reviewPreviewUrl, reviewQuad);
  }

  previewCanvas.addEventListener('pointerdown', event => {
    if (!reviewQuad) return;
    const rect = previewCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * previewCanvas.width / rect.width;
    const y = (event.clientY - rect.top) * previewCanvas.height / rect.height;
    dragIndex = reviewQuad.reduce((best, point, index) => {
      const distance = Math.hypot(point[0] - x, point[1] - y);
      return distance < Math.hypot(reviewQuad[best][0] - x, reviewQuad[best][1] - y) ? index : best;
    }, 0);
    previewCanvas.setPointerCapture(event.pointerId);
  });
  previewCanvas.addEventListener('pointermove', event => {
    if (dragIndex < 0 || !reviewQuad) return;
    const rect = previewCanvas.getBoundingClientRect();
    reviewQuad[dragIndex] = [
      Math.max(0, Math.min(previewCanvas.width - 1, (event.clientX - rect.left) * previewCanvas.width / rect.width)),
      Math.max(0, Math.min(previewCanvas.height - 1, (event.clientY - rect.top) * previewCanvas.height / rect.height)),
    ];
    saveActiveSceneQuad();
    redrawReview();
  });
  previewCanvas.addEventListener('pointerup', () => { dragIndex = -1; });

  if (surfacePicker) {
    surfacePicker.addEventListener('change', () => {
      activeSurfaceIndex = Math.max(0, Number(surfacePicker.value) || 0);
      const surfaces = activeSceneId == null ? [] : (sceneSurfaces.get(String(activeSceneId)) || []);
      if (surfaces[activeSurfaceIndex]) {
        reviewQuad = cloneQuad(surfaces[activeSurfaceIndex].quad);
        redrawReview();
      }
    });
  }

  async function pollJob(jobId) {
    activeJobId = jobId;
    localStorage.setItem('franchiseflow_job_id', jobId);
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || '读取任务状态失败');
    setStatus(`${job.status} · ${job.phase || ''}`.trim());
    if (job.status === 'queued' || job.status === 'running') {
      window.setTimeout(() => pollJob(jobId).catch(showError), 750);
      return;
    }
    if (job.status === 'needs_review') {
      showReview(job);
      return;
    }
    if (job.status === 'completed') {
      reviewPanel.hidden = true;
      setStatus('合成完成', 'badge badge-status');
      updateMetrics(job.summary || {}, job.initialization_mode, job);
      renderOutputs(job.outputs || []);
      return;
    }
    if (job.status === 'vision_failed') {
      setStatus('视觉定位失败', 'badge');
      throw new Error(job.error || '视觉模型没有返回可用候选');
    }
    throw new Error(job.error || '任务失败');
  }

  function showError(error) {
    setStatus('处理失败', 'badge');
    alert(`合成任务失败：${error.message}`);
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const files = branchInputs.map(input => input.files[0]).filter(Boolean);
    if (!videoInput.files[0]) return alert('请先选择门店视频');
    if (!files.length) return alert('请至少选择一张分支招牌图片');
    const formData = new FormData();
    formData.append('video', videoInput.files[0]);
    files.forEach(file => formData.append('branches', file));
    const targetHint = document.getElementById('target-hint-input');
    if (targetHint && targetHint.value.trim()) formData.append('target_hint', targetHint.value.trim());
    const manual = document.getElementById('mode-manual').checked;
    formData.append('use_manual_quad', manual ? 'true' : 'false');
    btnSubmit.disabled = true;
    btnSpinner.style.display = 'inline-block';
    btnText.textContent = '任务已提交，等待处理...';
    setStatus('queued');
    try {
      const response = await fetch('/api/jobs', { method: 'POST', body: formData });
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || `请求失败 (HTTP ${response.status})`);
      localStorage.setItem('franchiseflow_job_id', job.job_id);
      await pollJob(job.job_id);
    } catch (error) {
      showError(error);
    } finally {
      btnSubmit.disabled = false;
      btnSpinner.style.display = 'none';
      btnText.textContent = '开始跟踪与分支合成';
    }
  });

  reviewButton.addEventListener('click', async () => {
    if (!activeJobId || !reviewQuad) return;
    saveActiveSceneQuad();
    const payload = reviewScenes.length
      ? { scenes: reviewScenes.map(scene => ({
          scene_id: sceneKey(scene),
          surfaces: (sceneSurfaces.get(sceneKey(scene)) || [{ surface_id: `${sceneKey(scene)}_surface_001`, quad: sceneQuads.get(sceneKey(scene)) || centeredQuad() }])
            .map(surface => ({ surface_id: surface.surface_id, quad: surface.quad })),
        })), asset_bindings: [...sceneSurfaces.values()].flat().map(surface => surface.surface_id).length
          ? Array.from({ length: reviewBranchCount }, (_, index) => ({
              branch_index: index + 1,
              surface_ids: [...sceneSurfaces.values()].flat().map(surface => surface.surface_id),
              fit: 'cover',
            }))
          : undefined }
      : { quad: reviewQuad };
    reviewButton.disabled = true;
    try {
      const response = await fetch(`/api/jobs/${activeJobId}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || '提交人工确认失败');
      reviewPanel.hidden = true;
      await pollJob(activeJobId);
    } catch (error) {
      showError(error);
    } finally {
      reviewButton.disabled = false;
    }
  });

  btnLoadSample.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/demo_results');
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '样片不存在');
      setStatus('样片就绪 · 人工基线', 'badge badge-status');
      updateMetrics(data.summary || {}, data.initialization_mode || 'manual_fallback');
      drawSampleFrame('/static/sample_first_frame.jpg', data.vision_quad);
      renderOutputs(data.outputs || []);
      btnLoadSample.textContent = '已载入人工基线样片';
    } catch (error) {
      showError(error);
    }
  });

  const savedJobId = localStorage.getItem('franchiseflow_job_id');
  if (savedJobId) {
    pollJob(savedJobId).catch(showError);
  } else {
    // A browser refresh can lose localStorage while the durable job JSON still
    // contains a candidate waiting for review. Offer that latest task back to
    // the operator instead of forcing a re-upload.
    fetch('/api/jobs?status=needs_review&limit=1')
      .then(response => response.ok ? response.json() : { jobs: [] })
      .then(data => {
        const latest = Array.isArray(data.jobs) ? data.jobs[0] : null;
        if (latest && latest.job_id) {
          localStorage.setItem('franchiseflow_job_id', latest.job_id);
          return pollJob(latest.job_id);
        }
        return null;
      })
      .catch(() => null);
  }
});
