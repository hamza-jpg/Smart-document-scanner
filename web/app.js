/**
 * Smart Document Scanner - Interactive Canvas & API Client
 */

(function () {
  // DOM Elements
  const statusIndicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('statusText');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const canvasContainer = document.getElementById('canvasContainer');
  const canvas = document.getElementById('editorCanvas');
  const ctx = canvas.getContext('2d');
  const resetCornersBtn = document.getElementById('resetCornersBtn');
  const clearBtn = document.getElementById('clearBtn');
  const telemetryBar = document.getElementById('telemetryBar');
  const telemResolution = document.getElementById('telemResolution');
  const telemAlgo = document.getElementById('telemAlgo');
  const filterPills = document.querySelectorAll('.pill');
  const processBtn = document.getElementById('processBtn');
  const spinnerOverlay = document.getElementById('spinnerOverlay');
  const spinnerText = document.getElementById('spinnerText');
  const outputPlaceholder = document.getElementById('outputPlaceholder');
  const outputWrapper = document.getElementById('outputWrapper');
  const outputImage = document.getElementById('outputImage');
  const downloadBtn = document.getElementById('downloadBtn');
  const magnifier = document.getElementById('magnifier');
  const magnifierCanvas = document.getElementById('magnifierCanvas');
  const magCtx = magnifierCanvas.getContext('2d');

  // Application State
  let currentFile = null;
  let sourceImage = null;
  let corners = []; // [[x, y], ...] in original image coordinates
  let activeCornerIdx = -1;
  let selectedFilter = 'bw';
  let canvasScale = 1.0;
  const HANDLE_RADIUS = 12;
  const HIT_RADIUS = 28;

  // Initialize System Health Check
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        statusIndicator.classList.add('connected');
        statusText.textContent = 'Engine Ready';
      } else {
        throw new Error('Health check returned non-200');
      }
    } catch {
      statusIndicator.classList.remove('connected');
      statusText.textContent = 'Server Offline';
    }
  }
  checkHealth();
  setInterval(checkHealth, 15000);

  // File Upload Handlers
  browseBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUploadedFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUploadedFile(e.dataTransfer.files[0]);
    }
  });

  async function handleUploadedFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (JPG, PNG, WebP).');
      return;
    }

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        sourceImage = img;
        dropzone.classList.add('hidden');
        canvasContainer.classList.remove('hidden');
        telemetryBar.classList.remove('hidden');
        resetCornersBtn.disabled = false;
        clearBtn.disabled = false;
        processBtn.disabled = false;

        telemResolution.textContent = `${img.naturalWidth} x ${img.naturalHeight} px`;

        // Request automated corner detection from backend
        requestAutoDetection(file);
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  }

  // Auto Corner Detection via API
  async function requestAutoDetection(file) {
    showSpinner('Detecting document boundaries...');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Detection failed');
      }

      const data = await res.json();
      corners = data.corners;
      telemAlgo.textContent = data.algorithm;
    } catch (err) {
      console.warn('Auto detection warning:', err);
      // Fallback: 5% inset rectangle if API fails
      const w = sourceImage.naturalWidth;
      const h = sourceImage.naturalHeight;
      corners = [
        [w * 0.05, h * 0.05],
        [w * 0.95, h * 0.05],
        [w * 0.95, h * 0.95],
        [w * 0.05, h * 0.95]
      ];
      telemAlgo.textContent = 'default_inset';
    } finally {
      hideSpinner();
      resizeCanvas();
      drawCanvas();
    }
  }

  resetCornersBtn.addEventListener('click', () => {
    if (currentFile) {
      requestAutoDetection(currentFile);
    }
  });

  clearBtn.addEventListener('click', () => {
    currentFile = null;
    sourceImage = null;
    corners = [];
    dropzone.classList.remove('hidden');
    canvasContainer.classList.add('hidden');
    telemetryBar.classList.add('hidden');
    resetCornersBtn.disabled = true;
    clearBtn.disabled = true;
    processBtn.disabled = true;
    outputWrapper.classList.add('hidden');
    outputPlaceholder.classList.remove('hidden');
    fileInput.value = '';
  });

  // Canvas Sizing and Coordinates
  function resizeCanvas() {
    if (!sourceImage) return;

    const containerWidth = canvasContainer.clientWidth - 20;
    const maxHeight = window.innerHeight * 0.65;

    let targetWidth = containerWidth;
    let targetHeight = (sourceImage.naturalHeight / sourceImage.naturalWidth) * targetWidth;

    if (targetHeight > maxHeight) {
      targetHeight = maxHeight;
      targetWidth = (sourceImage.naturalWidth / sourceImage.naturalHeight) * targetHeight;
    }

    canvas.width = targetWidth;
    canvas.height = targetHeight;
    canvasScale = targetWidth / sourceImage.naturalWidth;
  }

  window.addEventListener('resize', () => {
    if (sourceImage) {
      resizeCanvas();
      drawCanvas();
    }
  });

  // Coordinate Conversion
  function imageToCanvas(pt) {
    return [pt[0] * canvasScale, pt[1] * canvasScale];
  }

  function canvasToImage(x, y) {
    return [x / canvasScale, y / canvasScale];
  }

  // Draw Interactive Canvas
  function drawCanvas() {
    if (!sourceImage) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw source photograph
    ctx.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);

    if (corners.length !== 4) return;

    const screenPts = corners.map(imageToCanvas);

    // 2. Draw semi-transparent document overlay
    ctx.beginPath();
    ctx.moveTo(screenPts[0][0], screenPts[0][1]);
    for (let i = 1; i < 4; i++) {
      ctx.lineTo(screenPts[i][0], screenPts[i][1]);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(99, 102, 241, 0.22)';
    ctx.fill();

    // 3. Draw glowing polygon edges
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#06b6d4';
    ctx.shadowColor = '#06b6d4';
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0; // reset shadow

    // 4. Draw corner handles
    const labels = ['TL', 'TR', 'BR', 'BL'];
    screenPts.forEach((pt, idx) => {
      const isSelected = (idx === activeCornerIdx);

      // Outer glow ring
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], isSelected ? HANDLE_RADIUS + 4 : HANDLE_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? 'rgba(6, 182, 212, 0.5)' : 'rgba(99, 102, 241, 0.35)';
      ctx.fill();

      // Inner solid circle
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], HANDLE_RADIUS - 3, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#4f46e5';
      ctx.stroke();

      // Label badge
      ctx.font = '600 10px Outfit, sans-serif';
      ctx.fillStyle = '#f8fafc';
      ctx.shadowColor = 'rgba(0,0,0,0.8)';
      ctx.shadowBlur = 4;
      ctx.fillText(labels[idx], pt[0] + 14, pt[1] - 8);
      ctx.shadowBlur = 0;
    });
  }

  // Magnifier Loupe Rendering
  function updateMagnifier(imgX, imgY, canvasX, canvasY) {
    const zoom = 2.2;
    const magW = magnifierCanvas.width;
    const magH = magnifierCanvas.height;

    magCtx.clearRect(0, 0, magW, magH);
    magCtx.drawImage(
      sourceImage,
      imgX - (magW / 2) / zoom,
      imgY - (magH / 2) / zoom,
      magW / zoom,
      magH / zoom,
      0,
      0,
      magW,
      magH
    );

    magnifier.style.left = `${canvasX}px`;
    magnifier.style.top = `${canvasY}px`;
    magnifier.classList.remove('hidden');
  }

  // Mouse & Touch Interaction
  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return [clientX - rect.left, clientY - rect.top];
  }

  function findClosestCorner(cx, cy) {
    if (!corners || corners.length !== 4) return -1;
    for (let i = 0; i < 4; i++) {
      const [sx, sy] = imageToCanvas(corners[i]);
      const dist = Math.hypot(cx - sx, cy - sy);
      if (dist <= HIT_RADIUS) {
        return i;
      }
    }
    return -1;
  }

  function onPointerDown(e) {
    const [cx, cy] = getCanvasCoords(e);
    activeCornerIdx = findClosestCorner(cx, cy);
    if (activeCornerIdx !== -1) {
      e.preventDefault();
      const [imgX, imgY] = corners[activeCornerIdx];
      updateMagnifier(imgX, imgY, cx, cy);
      drawCanvas();
    }
  }

  function onPointerMove(e) {
    const [cx, cy] = getCanvasCoords(e);

    if (activeCornerIdx !== -1) {
      e.preventDefault();
      // Clamp within original image boundaries
      const [imgX, imgY] = canvasToImage(cx, cy);
      const clampedX = Math.max(0, Math.min(sourceImage.naturalWidth, imgX));
      const clampedY = Math.max(0, Math.min(sourceImage.naturalHeight, imgY));

      corners[activeCornerIdx] = [clampedX, clampedY];
      updateMagnifier(clampedX, clampedY, cx, cy);
      drawCanvas();
    } else {
      // Hover cursor effect
      const hovered = findClosestCorner(cx, cy);
      canvas.style.cursor = hovered !== -1 ? 'pointer' : 'crosshair';
    }
  }

  function onPointerUp() {
    if (activeCornerIdx !== -1) {
      activeCornerIdx = -1;
      magnifier.classList.add('hidden');
      drawCanvas();
    }
  }

  // Canvas event listeners
  canvas.addEventListener('mousedown', onPointerDown);
  window.addEventListener('mousemove', onPointerMove);
  window.addEventListener('mouseup', onPointerUp);

  canvas.addEventListener('touchstart', onPointerDown, { passive: false });
  window.addEventListener('touchmove', onPointerMove, { passive: false });
  window.addEventListener('touchend', onPointerUp);

  // Filter Selector
  filterPills.forEach((pill) => {
    pill.addEventListener('click', () => {
      filterPills.forEach((p) => p.classList.remove('active'));
      pill.classList.add('active');
      selectedFilter = pill.getAttribute('data-filter');
    });
  });

  // Process / Scan Execution
  processBtn.addEventListener('click', async () => {
    if (!currentFile || corners.length !== 4) return;

    showSpinner('Rectifying perspective and applying enhancement filter...');
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('corners', JSON.stringify(corners));
    formData.append('filter_mode', selectedFilter);
    formData.append('output_format', 'jpg');

    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Scanning failed');
      }

      const blob = await res.blob();
      const imageUrl = URL.createObjectURL(blob);

      outputImage.src = imageUrl;
      downloadBtn.href = imageUrl;
      downloadBtn.download = `scan_${selectedFilter}_${Date.now()}.jpg`;

      outputPlaceholder.classList.add('hidden');
      outputWrapper.classList.remove('hidden');
    } catch (err) {
      alert(`Error during processing: ${err.message}`);
    } finally {
      hideSpinner();
    }
  });

  // Loading Overlay Utilities
  function showSpinner(text) {
    spinnerText.textContent = text;
    spinnerOverlay.classList.remove('hidden');
  }

  function hideSpinner() {
    spinnerOverlay.classList.add('hidden');
  }
})();
