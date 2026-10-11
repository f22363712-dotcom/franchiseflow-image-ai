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

  let branchInputs = [];
  let activeJobId = null;
  let reviewQuad = null;
  let dragIndex = -1;
  let reviewBackground = null;

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

  function drawVideoFirstFrame(file, quad = null) {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.playsInline = true;
    video.src = URL.createObjectURL(file);
    video.onloadeddata = () => { video.currentTime = 0; };
    video.onseeked = () => {
      placeholderPreview.style.display = 'none';
      previewCanvas.style.display = 'block';
      previewCanvas.width = video.videoWidth || 640;
      previewCanvas.height = video.videoHeight || 360;
      const ctx = previewCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
      reviewBackground = document.createElement('canvas');
      reviewBackground.width = previewCanvas.width;
      reviewBackground.height = previewCanvas.height;
      reviewBackground.getContext('2d').drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
      drawQuadOverlay(ctx, quad, Boolean(reviewPanel && !reviewPanel.hidden));
      URL.revokeObjectURL(video.src);
    };
  }

  function drawSampleFrame(url, quad) {
    const image = new Image();
    image.onload = () => {
      placeholderPreview.style.display = 'none';
      previewCanvas.style.display = 'block';
      previewCanvas.width = image.width;
      previewCanvas.height = image.height;
      const ctx = previewCanvas.getContext('2d');
      ctx.drawImage(image, 0, 0);
      reviewBackground = document.createElement('canvas');
      reviewBackground.width = image.width;
      reviewBackground.height = image.height;
      reviewBackground.getContext('2d').drawImage(image, 0, 0);
      drawQuadOverlay(ctx, quad, Boolean(reviewPanel && !reviewPanel.hidden));
    };
    image.src = url;
  }

  function redrawReview() {
    if (!reviewBackground) return;
    const ctx = previewCanvas.getContext('2d');
    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    ctx.drawImage(reviewBackground, 0, 0, previewCanvas.width, previewCanvas.height);
    drawQuadOverlay(ctx, reviewQuad, true);
  }

  function updateMetrics(summary, mode) {
    metricFrames.textContent = summary.frames ? `${summary.frames} 帧 (${summary.fps || 0} fps)` : '-';
    metricTime.textContent = summary.processing_seconds != null ? `${summary.processing_seconds}s` : '-';
    metricInliers.textContent = summary.median_inliers != null ? Math.round(summary.median_inliers) : '-';
    metricReproj.textContent = summary.reprojection_median_px != null ? `${summary.reprojection_median_px.toFixed(2)} px` : '-';
    metricFailed.textContent = summary.failed_frames ? `${summary.failed_frames.length} 帧` : '-';
    metricInitMode.textContent = mode || summary.initialization_mode || '-';
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
    reviewQuad = job.candidate && job.candidate.quad ? job.candidate.quad.map(point => point.slice()) : null;
    reviewPanel.hidden = false;
    reviewReasons.textContent = (job.validation && job.validation.reasons ? job.validation.reasons.join('、') : job.error || '候选没有通过三帧检查。');
    if (videoInput.files[0]) drawVideoFirstFrame(videoInput.files[0], reviewQuad);
    else if (job.sample_preview) drawSampleFrame(job.sample_preview, reviewQuad);
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
    const ctx = previewCanvas.getContext('2d');
    redrawReview();
  });
  previewCanvas.addEventListener('pointerup', () => { dragIndex = -1; });

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
      updateMetrics(job.summary || {}, job.initialization_mode);
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
    if (!videoInput.files[0]) return alert('请先选择 MP4 门店视频');
    if (!files.length) return alert('请至少选择一张分支招牌图片');
    const formData = new FormData();
    formData.append('video', videoInput.files[0]);
    files.forEach(file => formData.append('branches', file));
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
    reviewButton.disabled = true;
    try {
      const response = await fetch(`/api/jobs/${activeJobId}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ quad: reviewQuad }) });
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
      setStatus('样片就绪', 'badge badge-status');
      updateMetrics(data.summary || {}, data.initialization_mode || 'manual_fallback');
      drawSampleFrame('/static/sample_first_frame.jpg', data.vision_quad);
      renderOutputs(data.outputs || []);
      btnLoadSample.textContent = '已载入样例';
    } catch (error) {
      showError(error);
    }
  });

  const savedJobId = localStorage.getItem('franchiseflow_job_id');
  if (savedJobId) pollJob(savedJobId).catch(showError);
});
