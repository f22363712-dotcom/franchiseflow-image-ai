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
  const qualityReviewControls = document.getElementById('quality-review-controls');
  const qualityAcceptButton = document.getElementById('quality-accept');
  const qualityRejectButton = document.getElementById('quality-reject');
  const outputsGrid = document.getElementById('outputs-grid');
  const metricFrames = document.getElementById('metric-frames');
  const metricTime = document.getElementById('metric-time');
  const metricInliers = document.getElementById('metric-inliers');
  const metricReproj = document.getElementById('metric-reproj');
  const metricFailed = document.getElementById('metric-failed');
  const metricInitMode = document.getElementById('metric-init-mode');
  const metricScenes = document.getElementById('metric-scenes');
  const metricSurfaces = document.getElementById('metric-surfaces');
  const metricOutputContract = document.getElementById('metric-output-contract');
  const metricOcclusion = document.getElementById('metric-occlusion');
  const sceneReviewControls = document.getElementById('scene-review-controls');
  const scenePicker = document.getElementById('review-scene-picker');
  const surfacePicker = document.getElementById('review-surface-picker');
  const sceneReviewSummary = document.getElementById('scene-review-summary');
  const brandProfileInput = document.getElementById('brand-profile-input');
  const brandProfileOptions = document.getElementById('brand-profile-options');
  const brandProfileStatus = document.getElementById('brand-profile-status');
  const saveBrandProfileButton = document.getElementById('btn-save-brand-profile');
  const profileSaveName = document.getElementById('profile-save-name');
  const profileSaveId = document.getElementById('profile-save-id');
  const profileSaveLabels = document.getElementById('profile-save-labels');
  const profileSaveFit = document.getElementById('profile-save-fit');
  const profileSaveSurfacePolicy = document.getElementById('profile-save-surface-policy');
  const profileSaveScenePolicy = document.getElementById('profile-save-scene-policy');
  const profileSaveStatus = document.getElementById('brand-profile-save-status');
  const batchForm = document.getElementById('batch-form');
  const batchProfileSelect = document.getElementById('batch-profile-select');
  const batchProfileStatus = document.getElementById('batch-profile-status');
  const batchVideoInput = document.getElementById('batch-video-input');
  const batchVideoStatus = document.getElementById('batch-video-status');
  const batchSubmitButton = document.getElementById('btn-submit-batch');
  const batchSubmitSpinner = document.getElementById('batch-btn-spinner');
  const batchSubmitText = batchSubmitButton?.querySelector('.btn-text');
  const batchClearButton = document.getElementById('btn-clear-batch');
  const batchStatusBadge = document.getElementById('batch-status-badge');
  const batchResults = document.getElementById('batch-results');
  const batchSummary = document.getElementById('batch-summary');
  const batchJobs = document.getElementById('batch-jobs');

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
  let brandProfiles = new Map();
  let activeBatchId = null;
  let batchPollToken = 0;
  let batchVideoNames = [];

  const batchStatusLabels = {
    queued: '排队中',
    running: '处理中',
    completed: '已完成',
    needs_review: '需要复核',
    partial_failed: '部分失败',
    failed: '失败',
    missing: '任务缺失',
    vision_failed: '视觉定位失败',
  };

  function batchStatusLabel(status) {
    return batchStatusLabels[String(status || '')] || String(status || '未知');
  }

  function setBatchStatus(text, className = 'badge badge-tech') {
    if (!batchStatusBadge) return;
    batchStatusBadge.textContent = text;
    batchStatusBadge.className = className;
  }

  function renderBatchProfileOptions() {
    if (!batchProfileSelect) return;
    const current = batchProfileSelect.value;
    batchProfileSelect.replaceChildren();
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = brandProfiles.size ? '选择品牌配置…' : '暂无可用品牌配置';
    batchProfileSelect.appendChild(placeholder);
    [...brandProfiles.values()].forEach(profile => {
      const option = document.createElement('option');
      option.value = String(profile.profile_id);
      option.textContent = `${profile.brand_name || profile.profile_id} · ${(profile.branch_labels || []).length} 个分店`;
      batchProfileSelect.appendChild(option);
    });
    if (current && brandProfiles.has(current)) batchProfileSelect.value = current;
  }

  function renderBrandProfileOptions() {
    if (brandProfileOptions) brandProfileOptions.replaceChildren(...[...brandProfiles.values()].map(profile => {
      const option = document.createElement('option');
      option.value = String(profile.profile_id);
      option.label = `${profile.brand_name || profile.profile_id} · ${(profile.branch_labels || []).length} 个分店`;
      return option;
    }));
    renderBatchProfileOptions();
  }

  async function loadBrandProfiles() {
    if (!brandProfileOptions) return;
    try {
      const response = await fetch('/api/brand-profiles');
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || '读取品牌配置失败');
      brandProfiles = new Map((payload.profiles || []).map(profile => [String(profile.profile_id), profile]));
      renderBrandProfileOptions();
      if (brandProfileStatus) brandProfileStatus.textContent = brandProfiles.size ? `已载入 ${brandProfiles.size} 个本地配置，可从列表选择。` : '暂无本地配置，可直接上传分支图片。';
      if (batchProfileStatus) batchProfileStatus.textContent = brandProfiles.size ? `已载入 ${brandProfiles.size} 个品牌配置。` : '请先在单视频表单中保存一个品牌配置。';
    } catch (error) {
      if (brandProfileStatus) brandProfileStatus.textContent = '品牌配置列表暂时不可用，可继续直接上传分支图片。';
      if (batchProfileStatus) batchProfileStatus.textContent = '品牌配置列表暂时不可用，批量提交需要稍后重试。';
    }
  }

  function updateBrandProfileStatus() {
    if (!brandProfileStatus || !brandProfileInput) return;
    const profile = brandProfiles.get(brandProfileInput.value.trim());
    if (profile) {
      const policy = profile.surface_policy === 'all_surfaces' ? '全平面映射' : '多平面需复核';
      brandProfileStatus.textContent = `${profile.brand_name || profile.profile_id} · ${(profile.branch_labels || []).join('、')} · ${policy}`;
    } else if (brandProfileInput.value.trim()) {
      brandProfileStatus.textContent = '未找到这个配置 ID，请从列表选择或检查拼写。';
    }
  }

  brandProfileInput && brandProfileInput.addEventListener('input', updateBrandProfileStatus);
  loadBrandProfiles();

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

  function clearUploadedBranches() {
    branchInputs.forEach(input => {
      input.value = '';
      const name = input.closest('.branch-upload-box')?.querySelector('.branch-file-name');
      if (name) name.textContent = `选择图片 ${input.closest('.branch-upload-box')?.dataset.index || ''}`;
    });
  }

  function setProfileSaveStatus(text, tone = 'muted') {
    if (!profileSaveStatus) return;
    profileSaveStatus.textContent = text;
    profileSaveStatus.style.color = tone === 'error' ? 'var(--danger)' : tone === 'success' ? 'var(--success)' : '';
  }

  async function saveCurrentBrandProfile() {
    const files = branchInputs.map(input => input.files[0]).filter(Boolean);
    const brandName = profileSaveName ? profileSaveName.value.trim() : '';
    if (!brandName) {
      setProfileSaveStatus('请先填写品牌名称。', 'error');
      profileSaveName?.focus();
      return;
    }
    if (brandProfileInput?.value.trim()) {
      setProfileSaveStatus('当前已选品牌配置，请先清空配置 ID，再保存新的分支素材。', 'error');
      return;
    }
    if (!files.length) {
      setProfileSaveStatus('请先上传至少一张分支图片。', 'error');
      return;
    }
    const labels = (profileSaveLabels?.value || '')
      .split(/[,，]/)
      .map(label => label.trim())
      .filter(Boolean);
    if (labels.length && labels.length !== files.length) {
      setProfileSaveStatus(`分支标签有 ${labels.length} 个，但图片有 ${files.length} 张，请逐一对应。`, 'error');
      return;
    }
    const payload = new FormData();
    payload.append('brand_name', brandName);
    if (profileSaveId?.value.trim()) payload.append('profile_id', profileSaveId.value.trim());
    const targetHint = document.getElementById('target-hint-input');
    if (targetHint?.value.trim()) payload.append('target_hint', targetHint.value.trim());
    if (labels.length) payload.append('branch_labels', JSON.stringify(labels));
    payload.append('default_fit', profileSaveFit?.value || 'cover');
    payload.append('surface_policy', profileSaveSurfacePolicy?.value || 'review');
    payload.append('scene_reuse_policy', profileSaveScenePolicy?.value || 'vision_each_scene');
    files.forEach(file => payload.append('branches', file));
    saveBrandProfileButton.disabled = true;
    setProfileSaveStatus('正在保存品牌图片和策略…');
    try {
      const response = await fetch('/api/brand-profiles', { method: 'POST', body: payload });
      const profile = await response.json();
      if (!response.ok) throw new Error(profile.error || `保存失败（HTTP ${response.status}）`);
      brandProfiles.set(String(profile.profile_id), profile);
      renderBrandProfileOptions();
      if (brandProfileInput) brandProfileInput.value = String(profile.profile_id);
      updateBrandProfileStatus();
      clearUploadedBranches();
      setProfileSaveStatus(`已保存 ${profile.brand_name || profile.profile_id}，现在可以直接提交视频任务。`, 'success');
    } catch (error) {
      setProfileSaveStatus(error.message || '保存品牌配置失败。', 'error');
    } finally {
      saveBrandProfileButton.disabled = false;
    }
  }

  saveBrandProfileButton && saveBrandProfileButton.addEventListener('click', saveCurrentBrandProfile);

  function updateBatchVideoStatus() {
    if (!batchVideoStatus || !batchVideoInput) return;
    const files = [...(batchVideoInput.files || [])];
    batchVideoStatus.textContent = files.length
      ? `已选择 ${files.length} 条视频：${files.map(file => file.name).join('、')}`
      : '尚未选择视频。';
  }

  function updateBatchProfileStatus() {
    if (!batchProfileStatus || !batchProfileSelect) return;
    const profile = brandProfiles.get(batchProfileSelect.value);
    batchProfileStatus.textContent = profile
      ? `${profile.brand_name || profile.profile_id} · ${(profile.branch_labels || []).join('、') || '未设置分店标签'} · ${(profile.branch_assets || []).length || (profile.branch_files || []).length || 0} 个素材`
      : (brandProfiles.size ? '品牌配置会提供分店素材和识别策略。' : '请先在单视频表单中保存一个品牌配置。');
  }

  function batchOutputLinks(job) {
    const outputs = Array.isArray(job.outputs) ? job.outputs : [];
    return outputs.map((url, index) => {
      const link = document.createElement('a');
      link.className = 'btn btn-outline';
      link.href = url;
      link.target = '_blank';
      link.rel = 'noopener';
      link.download = `job_${job.job_id}_branch_${index + 1}.mp4`;
      link.textContent = `下载成片 ${index + 1}`;
      return link;
    });
  }

  function openBatchJob(jobId) {
    if (!jobId) return;
    activeJobId = String(jobId);
    localStorage.setItem('franchiseflow_job_id', activeJobId);
    pollJob(activeJobId)
      .then(() => {
        document.getElementById('review-panel')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        if (activeBatchId) {
          batchPollToken += 1;
          return pollBatch(activeBatchId, batchPollToken);
        }
        return null;
      })
      .catch(showError);
  }

  function renderBatch(batch) {
    if (!batchResults || !batchJobs) return;
    batchResults.hidden = false;
    const counts = batch.counts || {};
    const total = Number(counts.total || batch.requested_count || (batch.jobs || []).length || 0);
    batchSummary.textContent = `批次 ${batch.batch_id || ''} · ${batchStatusLabel(batch.status)} · 已完成 ${counts.completed || 0}/${total} · 处理中 ${counts.running || 0} · 待复核 ${counts.needs_review || 0} · 失败 ${counts.failed || 0}`;
    batchJobs.replaceChildren();
    (Array.isArray(batch.jobs) ? batch.jobs : []).forEach((job, index) => {
      const card = document.createElement('article');
      card.className = 'batch-job-card';
      const header = document.createElement('div');
      header.className = 'batch-job-header';
      const title = document.createElement('div');
      title.className = 'batch-job-title';
      title.textContent = job.video_file || batchVideoNames[index] || `视频 ${index + 1}`;
      const badge = document.createElement('span');
      badge.className = `batch-job-status status-${String(job.status || 'unknown')}`;
      badge.textContent = batchStatusLabel(job.status);
      header.append(title, badge);
      card.appendChild(header);
      const id = document.createElement('div');
      id.className = 'batch-job-id';
      id.textContent = job.job_id ? `任务 ${job.job_id}` : '任务 ID 不可用';
      card.appendChild(id);
      if (job.phase) {
        const phase = document.createElement('div');
        phase.className = 'mode-tip';
        phase.textContent = `阶段：${job.phase}`;
        card.appendChild(phase);
      }
      if (job.error) {
        const error = document.createElement('p');
        error.className = 'batch-job-error';
        error.textContent = job.error;
        card.appendChild(error);
      }
      const actions = document.createElement('div');
      actions.className = 'batch-job-actions';
      if (job.status === 'needs_review') {
        const review = document.createElement('button');
        review.type = 'button';
        review.className = 'btn btn-secondary';
        review.textContent = '打开人工复核';
        review.addEventListener('click', () => openBatchJob(job.job_id));
        actions.appendChild(review);
      }
      batchOutputLinks(job).forEach(link => actions.appendChild(link));
      if (actions.children.length) card.appendChild(actions);
      batchJobs.appendChild(card);
    });
  }

  function batchIsTerminal(batch) {
    const jobs = Array.isArray(batch.jobs) ? batch.jobs : [];
    if (!jobs.length) return ['failed', 'completed', 'partial_failed'].includes(batch.status);
    return jobs.every(job => ['completed', 'needs_review', 'failed', 'vision_failed', 'missing'].includes(job.status));
  }

  async function pollBatch(batchId, token = batchPollToken) {
    const response = await fetch(`/api/batches/${encodeURIComponent(batchId)}`);
    const batch = await response.json();
    if (!response.ok) throw new Error(batch.error || '读取批量任务状态失败');
    if (token !== batchPollToken) return batch;
    activeBatchId = batchId;
    renderBatch(batch);
    setBatchStatus(batchStatusLabel(batch.status), batch.status === 'completed' ? 'badge badge-status' : 'badge badge-tech');
    if (!batchIsTerminal(batch)) {
      await new Promise(resolve => window.setTimeout(resolve, 1000));
      return pollBatch(batchId, token);
    }
    return batch;
  }

  function showBatchError(error) {
    setBatchStatus('批量任务失败', 'badge');
    if (batchResults) batchResults.hidden = false;
    if (batchSummary) batchSummary.textContent = error.message || '批量任务状态读取失败。';
  }

  batchVideoInput && batchVideoInput.addEventListener('change', updateBatchVideoStatus);
  batchProfileSelect && batchProfileSelect.addEventListener('change', updateBatchProfileStatus);
  batchClearButton && batchClearButton.addEventListener('click', () => {
    batchPollToken += 1;
    activeBatchId = null;
    batchVideoNames = [];
    if (batchVideoInput) batchVideoInput.value = '';
    updateBatchVideoStatus();
    if (batchResults) batchResults.hidden = true;
    if (batchJobs) batchJobs.replaceChildren();
    if (batchSummary) batchSummary.textContent = '';
    setBatchStatus('等待提交');
    localStorage.removeItem('franchiseflow_batch_id');
  });

  batchForm && batchForm.addEventListener('submit', async event => {
    event.preventDefault();
    const profileId = batchProfileSelect?.value.trim() || '';
    const files = [...(batchVideoInput?.files || [])];
    if (!profileId) return showBatchError(new Error('请选择一个品牌配置。'));
    if (!files.length) return showBatchError(new Error('请至少选择一条门店视频。'));
    if (files.length > 8) return showBatchError(new Error('一次最多提交 8 条视频。'));
    const payload = new FormData();
    payload.append('brand_profile_id', profileId);
    batchVideoNames = files.map(file => file.name);
    files.forEach(file => payload.append('videos', file));
    batchSubmitButton.disabled = true;
    if (batchSubmitSpinner) batchSubmitSpinner.style.display = 'inline-block';
    if (batchSubmitText) batchSubmitText.textContent = '正在提交批次…';
    setBatchStatus('提交中');
    try {
      const response = await fetch('/api/batches', { method: 'POST', body: payload });
      const batch = await response.json();
      if (!response.ok) throw new Error(batch.error || `批量请求失败（HTTP ${response.status}）`);
      activeBatchId = batch.batch_id;
      batchPollToken += 1;
      localStorage.setItem('franchiseflow_batch_id', activeBatchId);
      renderBatch(batch);
      await pollBatch(activeBatchId, batchPollToken);
    } catch (error) {
      showBatchError(error);
    } finally {
      batchSubmitButton.disabled = false;
      if (batchSubmitSpinner) batchSubmitSpinner.style.display = 'none';
      if (batchSubmitText) batchSubmitText.textContent = '提交批量任务';
    }
  });

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
    const transition = scene.transition_type && scene.transition_type !== 'none' ? ` · 转场 ${scene.transition_type}` : '';
    sceneReviewSummary.textContent = `${sceneRange(scene)}${transition} · 初始化 ${sceneInitMode(scene)} · ${sceneStatus(scene)} · ${surfaceCount || 0} 个平面${surfaceStatus}`;
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
    if (metricOutputContract) {
      const checks = Array.isArray(summary.output_validation) ? summary.output_validation : [];
      metricOutputContract.textContent = checks.length && checks.every(item => item.validated) ? `已验收 ${checks.length} 条` : (mode === 'manual_fallback' ? '样片基线' : '-');
    }
    if (metricOcclusion) {
      metricOcclusion.textContent = summary.occlusion_hint_count != null ? `${summary.occlusion_hint_count} 帧` : '-';
    }
  }

  function renderOutputs(urls, labels = []) {
    outputsGrid.innerHTML = '';
    (urls || []).forEach((url, index) => {
      const label = labels[index] || `分支门店 ${index + 1}`;
      const card = document.createElement('div');
      card.className = 'output-card';
      card.innerHTML = `<div class="output-card-header"><span class="branch-tag">Branch ${index + 1}</span><span class="branch-title"></span></div><div class="video-container"><video controls playsinline loop></video></div><div class="output-actions"><a class="btn btn-outline" download>下载成片 MP4</a></div>`;
      card.querySelector('.branch-title').textContent = label;
      const player = card.querySelector('video');
      const link = card.querySelector('a');
      player.src = url;
      player.addEventListener('error', () => {
        card.querySelector('.branch-title').textContent += '（浏览器无法播放）';
      });
      link.href = url;
      link.download = `branch_${index + 1}_${label}.mp4`;
      outputsGrid.appendChild(card);
    });
  }

  function showReview(job) {
    reviewPanel.hidden = false;
    reviewBranchCount = Math.max(1, Number(job.branch_count) || 1);
    reviewPreviewUrl = job.sample_preview || job.preview || null;
    const qualityReview = job.phase === 'quality_review' && job.quality_gate && job.quality_gate.passed === false;
    if (qualityReview) {
      if (qualityReviewControls) qualityReviewControls.hidden = false;
      if (reviewButton) reviewButton.hidden = true;
      if (sceneReviewControls) sceneReviewControls.hidden = true;
      const gate = job.quality_gate;
      reviewReasons.textContent = `跟踪质量需要复核：失败帧 ${gate.failed_frames || 0}，失败比例 ${(Number(gate.failed_fraction || 0) * 100).toFixed(1)}%，阈值 ${(Number(gate.max_failed_fraction || 0) * 100).toFixed(1)}%。`;
      renderOutputs(job.outputs || [], job.branch_labels || (job.summary && job.summary.branch_labels) || []);
      return;
    }
    if (qualityReviewControls) qualityReviewControls.hidden = true;
    if (reviewButton) reviewButton.hidden = false;
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
      renderOutputs(job.outputs || [], job.branch_labels || (job.summary && job.summary.branch_labels) || []);
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
    const brandProfileId = brandProfileInput ? brandProfileInput.value.trim() : '';
    if (!brandProfileId && !files.length) return alert('请至少选择一张分支招牌图片，或填写品牌配置 ID');
    if (brandProfileId && files.length) return alert('使用品牌配置时请清空本次上传的分支图片');
    const formData = new FormData();
    formData.append('video', videoInput.files[0]);
    if (!brandProfileId) files.forEach(file => formData.append('branches', file));
    if (brandProfileId) formData.append('brand_profile_id', brandProfileId);
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

  async function submitQualityDecision(decision) {
    if (!activeJobId) return;
    qualityAcceptButton && (qualityAcceptButton.disabled = true);
    qualityRejectButton && (qualityRejectButton.disabled = true);
    try {
      const response = await fetch(`/api/jobs/${activeJobId}/quality-review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision }),
      });
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || '提交质量复核失败');
      await pollJob(activeJobId);
    } catch (error) {
      showError(error);
    } finally {
      qualityAcceptButton && (qualityAcceptButton.disabled = false);
      qualityRejectButton && (qualityRejectButton.disabled = false);
    }
  }

  if (qualityAcceptButton) qualityAcceptButton.addEventListener('click', () => submitQualityDecision('accept'));
  if (qualityRejectButton) qualityRejectButton.addEventListener('click', () => submitQualityDecision('reject'));

  btnLoadSample.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/demo_results');
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '样片不存在');
      setStatus('样片就绪 · 人工基线', 'badge badge-status');
      updateMetrics(data.summary || {}, data.initialization_mode || 'manual_fallback');
      drawSampleFrame('/static/sample_first_frame.jpg', data.vision_quad);
      renderOutputs(data.outputs || [], data.branch_labels || (data.summary && data.summary.branch_labels) || []);
      btnLoadSample.textContent = '已载入人工基线样片';
    } catch (error) {
      showError(error);
    }
  });

  const savedJobId = localStorage.getItem('franchiseflow_job_id');
  const savedBatchId = localStorage.getItem('franchiseflow_batch_id');
  if (savedBatchId && !savedJobId) {
    activeBatchId = savedBatchId;
    batchPollToken += 1;
    pollBatch(savedBatchId, batchPollToken).catch(showBatchError);
  } else if (savedJobId) {
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
