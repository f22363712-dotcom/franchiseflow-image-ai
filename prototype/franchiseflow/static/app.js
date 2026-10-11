document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('render-form');
  const videoInput = document.getElementById('video-input');
  const videoFileName = document.getElementById('video-file-name');
  const branchInputs = [
    document.getElementById('branch-input-1'),
    document.getElementById('branch-input-2'),
    document.getElementById('branch-input-3')
  ];
  const branchNames = [
    document.getElementById('branch-name-1'),
    document.getElementById('branch-name-2'),
    document.getElementById('branch-name-3')
  ];

  const btnSubmit = document.getElementById('btn-submit');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnText = btnSubmit.querySelector('.btn-text');
  const btnLoadSample = document.getElementById('btn-load-sample');

  const quadStatusBadge = document.getElementById('quad-status-badge');
  const canvasWrapper = document.getElementById('canvas-wrapper');
  const placeholderPreview = document.getElementById('placeholder-preview');
  const previewCanvas = document.getElementById('preview-canvas');

  const metricFrames = document.getElementById('metric-frames');
  const metricTime = document.getElementById('metric-time');
  const metricInliers = document.getElementById('metric-inliers');
  const metricReproj = document.getElementById('metric-reproj');
  const metricFailed = document.getElementById('metric-failed');
  const metricInitMode = document.getElementById('metric-init-mode');

  const videoPlayers = [
    document.getElementById('video-player-1'),
    document.getElementById('video-player-2'),
    document.getElementById('video-player-3')
  ];
  const downloadLinks = [
    document.getElementById('download-link-1'),
    document.getElementById('download-link-2'),
    document.getElementById('download-link-3')
  ];

  // Update file name display
  videoInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      videoFileName.textContent = e.target.files[0].name;
      drawVideoFirstFrame(e.target.files[0]);
    }
  });

  branchInputs.forEach((input, index) => {
    input.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        branchNames[index].textContent = e.target.files[0].name;
      }
    });
  });

  // Extract first frame of video for client-side preview
  function drawVideoFirstFrame(videoFile, quad = null) {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.src = URL.createObjectURL(videoFile);
    video.muted = true;
    video.playsInline = true;

    video.onloadeddata = () => {
      video.currentTime = 0;
    };

    video.onseeked = () => {
      placeholderPreview.style.display = 'none';
      previewCanvas.style.display = 'block';

      const ctx = previewCanvas.getContext('2d');
      previewCanvas.width = video.videoWidth || 640;
      previewCanvas.height = video.videoHeight || 360;

      ctx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);

      if (quad && quad.length === 4) {
        drawQuadOverlay(ctx, quad);
      }
      URL.revokeObjectURL(video.src);
    };
  }

  function drawQuadOverlay(ctx, quad) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#06b6d4';
    ctx.fillStyle = 'rgba(6, 182, 212, 0.2)';

    ctx.beginPath();
    ctx.moveTo(quad[0][0], quad[0][1]);
    for (let i = 1; i < 4; i++) {
      ctx.lineTo(quad[i][0], quad[i][1]);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.fill();

    const labels = ['TL', 'TR', 'BR', 'BL'];
    quad.forEach((pt, idx) => {
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(`${labels[idx]} (${Math.round(pt[0])}, ${Math.round(pt[1])})`, pt[0] + 8, pt[1] - 4);
    });
  }

  // Sample loader helper
  btnLoadSample.addEventListener('click', () => {
    alert('已提供真实样片位于 spikes/franchiseflow/fixtures/ 与 assets/，可在表单中直接选择 real_storefront.mp4 及 riverside.png、downtown.png、northpark.png。');
  });

  // Submit Handler
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!videoInput.files[0]) {
      alert('请先选择 MP4 门店视频');
      return;
    }
    for (let i = 0; i < 3; i++) {
      if (!branchInputs[i].files[0]) {
        alert(`请选择分支 ${i + 1} 的招牌图片`);
        return;
      }
    }

    const isManual = document.getElementById('mode-manual').checked;

    const formData = new FormData();
    formData.append('video', videoInput.files[0]);
    formData.append('branch_1', branchInputs[0].files[0]);
    formData.append('branch_2', branchInputs[1].files[0]);
    formData.append('branch_3', branchInputs[2].files[0]);
    formData.append('use_manual_quad', isManual ? 'true' : 'false');

    // UI Loading State
    btnSubmit.disabled = true;
    btnSpinner.style.display = 'inline-block';
    btnText.textContent = isManual ? '正在使用手动四角跟踪合成...' : '正在调用 GLM 视觉定位与跟踪合成...';
    quadStatusBadge.textContent = '处理中...';
    quadStatusBadge.className = 'badge badge-tech';

    try {
      const response = await fetch('/api/render', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `请求失败 (HTTP ${response.status})`);
      }

      // Success
      quadStatusBadge.textContent = '合成完成';
      quadStatusBadge.className = 'badge badge-status';

      // Update metrics
      const summary = data.summary || {};
      metricFrames.textContent = `${summary.frames || 0} 帧 (${summary.fps || 24} fps)`;
      metricTime.textContent = `${summary.processing_seconds || 0}s`;
      metricInliers.textContent = summary.median_inliers ? Math.round(summary.median_inliers) : 'N/A';
      metricReproj.textContent = summary.reprojection_median_px ? `${summary.reprojection_median_px.toFixed(2)} px` : 'N/A';
      metricFailed.textContent = summary.failed_frames ? `${summary.failed_frames.length} 帧 (${((summary.failed_fraction || 0) * 100).toFixed(1)}%)` : '0';
      metricInitMode.textContent = data.initialization_mode === 'vision_model' ? 'GLM-5.3-flash 视觉模型' : '手动基线兜底 (Manual)';

      // Draw quad on canvas
      if (data.vision_quad && videoInput.files[0]) {
        drawVideoFirstFrame(videoInput.files[0], data.vision_quad);
      }

      // Populate video players
      if (data.outputs && data.outputs.length === 3) {
        data.outputs.forEach((url, i) => {
          videoPlayers[i].src = url;
          videoPlayers[i].load();
          downloadLinks[i].href = url;
        });
      }

    } catch (err) {
      console.error(err);
      quadStatusBadge.textContent = '处理失败';
      quadStatusBadge.className = 'badge';
      quadStatusBadge.style.color = '#ef4444';
      quadStatusBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';

      alert(`合成任务失败: ${err.message}\n\n提示：若为视觉网关 403 错误，可选择“手动基线兜底”继续本地高质量几何跟踪。`);
    } finally {
      btnSubmit.disabled = false;
      btnSpinner.style.display = 'none';
      btnText.textContent = '开始跟踪与分支合成';
    }
  });
});
