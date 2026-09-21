/**
 * Dashboard & Video Pipeline Controller (High Responsiveness & Stability)
 * Provides instant button feedback, asynchronous plot generation,
 * smoothed steering indicator, and multi-terrain demo loading.
 */

class DashboardController {
  constructor() {
    this.ws = null;
    this.map = null;
    this.isPlaying = false;
    this.currentVideoSource = 'None';
    this.lastAction = '';
    this.smoothedDisplaySteer = 0.0;

    this.initElements();
    this.initTrajectoryMap();
    this.initWebSocket();
    this.bindEvents();
    if (this.sourceLabel) {
      this.sourceLabel.innerText = 'Source: Please upload a video file or connect a webcam';
    }
  }

  initElements() {
    // Buttons
    this.btnPlay = document.getElementById('btn-play');
    this.btnStep = document.getElementById('btn-step');
    this.btnRestart = document.getElementById('btn-restart');
    this.btnPlot = document.getElementById('btn-plot');

    // Source buttons
    this.btnUpload = document.getElementById('btn-upload');
    this.fileInput = document.getElementById('video-file-input');
    this.btnDemo = document.getElementById('btn-demo');
    this.btnDemoOffroad = document.getElementById('btn-demo-offroad');
    this.btnWebcam = document.getElementById('btn-webcam');
    this.sourceLabel = document.getElementById('source-label');

    // Feeds
    this.feedMain = document.getElementById('feed-main-video');
    this.feedTraversable = document.getElementById('feed-traversable');
    this.feedOpticalFlow = document.getElementById('feed-optical-flow');
    this.actionBanner = document.getElementById('action-banner');

    // Telemetry displays
    this.statSpeed = document.getElementById('stat-speed');
    this.statSteer = document.getElementById('stat-steer');
    this.statDist = document.getElementById('stat-dist');
    this.statObsCount = document.getElementById('stat-obs-count');
    this.statNearestObs = document.getElementById('stat-nearest-obs');
    this.statVoPose = document.getElementById('stat-vo-pose');
    this.statFeatures = document.getElementById('stat-features');

    // Steering dial
    this.steeringIndicator = document.getElementById('steering-indicator');

    // Badges & Console
    this.wsDot = document.getElementById('ws-dot');
    this.wsStatusText = document.getElementById('ws-status-text');
    this.consoleOutput = document.getElementById('console-output');

    // Modal
    this.plotModal = document.getElementById('plot-modal');
    this.modalClose = document.getElementById('modal-close');
    this.plotImg = document.getElementById('plot-modal-img');
  }

  initTrajectoryMap() {
    this.map = new TrajectoryMap('trajectory-canvas');
  }

  initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/ugv`;

    this.logConsole(`[SYSTEM] Connecting to backend at ${wsUrl}...`);
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.wsDot.classList.add('connected');
      this.wsStatusText.innerText = 'ONLINE (WS Connected)';
      this.logConsole('[SYSTEM] Connected to Python Server. Ready.');
    };

    this.ws.onclose = () => {
      this.wsDot.classList.remove('connected');
      this.wsStatusText.innerText = 'DISCONNECTED';
      this.logConsole('[SYSTEM] WebSocket closed. Retrying in 2s...');
      setTimeout(() => this.initWebSocket(), 2000);
    };

    this.ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };

    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      this.handleServerMessage(msg);
    };
  }

  bindEvents() {
    // Immediate response button handlers
    this.btnPlay.addEventListener('click', () => this.togglePlay());
    this.btnStep.addEventListener('click', () => this.stepForward());
    this.btnRestart.addEventListener('click', () => this.restartVideo());
    this.btnPlot.addEventListener('click', () => this.requestPlot());

    // Source buttons
    this.btnUpload.addEventListener('click', () => this.fileInput.click());
    this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
    if (this.btnDemo) {
      this.btnDemo.addEventListener('click', () => this.loadDemoVideo());
    }
    if (this.btnDemoOffroad) {
      this.btnDemoOffroad.addEventListener('click', () => this.loadDemoOffroadVideo());
    }
    this.btnWebcam.addEventListener('click', () => this.openWebcam());

    // Modal Close
    this.modalClose.addEventListener('click', () => {
      this.plotModal.classList.remove('active');
    });

    // Close modal on click outside content
    this.plotModal.addEventListener('click', (e) => {
      if (e.target === this.plotModal) {
        this.plotModal.classList.remove('active');
      }
    });
  }

  togglePlay() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      alert('WebSocket not connected. Ensure run_server.py is active.');
      return;
    }

    this.isPlaying = !this.isPlaying;
    // Immediate UI feedback
    this.btnPlay.innerText = this.isPlaying ? '⏸ Pause' : '▶ Play';

    if (this.isPlaying) {
      this.ws.send(JSON.stringify({ type: 'PLAY' }));
      this.logConsole('[PLAYBACK] Started video stream.');
    } else {
      this.ws.send(JSON.stringify({ type: 'PAUSE' }));
      this.logConsole('[PLAYBACK] Paused.');
    }
  }

  stepForward() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.isPlaying = false;
      this.btnPlay.innerText = '▶ Play';
      this.ws.send(JSON.stringify({ type: 'STEP' }));
    }
  }

  restartVideo() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.isPlaying = false;
      this.btnPlay.innerText = '▶ Play';
      this.smoothedDisplaySteer = 0.0;
      this.ws.send(JSON.stringify({ type: 'RESTART' }));
      this.map.clear();
      this.logConsole('[PLAYBACK] Video reset to frame 0.');
    }
  }

  async requestPlot() {
    // Immediate visual feedback on click
    const originalText = this.btnPlot.innerText;
    this.btnPlot.innerText = '⏳ Generating Plot...';
    this.btnPlot.disabled = true;

    try {
      // Pause playback if running so plot captures up to current moment
      if (this.isPlaying) {
        this.togglePlay();
      }

      const resp = await fetch('/api/generate_plot');
      const data = await resp.json();

      if (data.plot_image) {
        this.showPlotModal(data.plot_image);
      } else {
        alert('No trajectory data recorded yet. Play the video first.');
      }
    } catch (err) {
      console.error('Plot error:', err);
      this.logConsole(`[ERROR] Failed to fetch plot: ${err.message}`);
    } finally {
      this.btnPlot.innerText = originalText;
      this.btnPlot.disabled = false;
    }
  }

  async handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    this.sourceLabel.innerText = `Uploading ${file.name}...`;
    this.logConsole(`[UPLOAD] Uploading ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const resp = await fetch('/api/upload_video', {
        method: 'POST',
        body: formData,
      });
      const data = await resp.json();

      if (data.status === 'success') {
        this.currentVideoSource = file.name;
        this.sourceLabel.innerText = `File: ${file.name} (${data.metadata.total_frames} frames)`;
        this.map.clear();
        this.isPlaying = false;
        this.btnPlay.innerText = '▶ Play';
        this.smoothedDisplaySteer = 0.0;
        this.logConsole(`[UPLOAD SUCCESS] Loaded ${file.name}. Click Play to start.`);
        this.stepForward();
      } else {
        alert('Failed to load video file.');
      }
    } catch (err) {
      console.error('Upload error:', err);
      this.logConsole(`[UPLOAD ERROR] ${err.message}`);
    }
  }

  async loadDemoVideo() {
    this.sourceLabel.innerText = 'Loading paved road demo video...';
    try {
      const resp = await fetch('/api/load_demo', { method: 'POST' });
      const data = await resp.json();
      if (data.status === 'success') {
        this.currentVideoSource = 'Demo Paved Road (White Lines)';
        this.sourceLabel.innerText = `Demo: Paved Road (${data.metadata.total_frames} frames)`;
        this.map.clear();
        this.isPlaying = false;
        this.btnPlay.innerText = '▶ Play';
        this.smoothedDisplaySteer = 0.0;
        this.logConsole('[SOURCE] Loaded paved road demo. Click Play to start.');
        this.stepForward();
      }
    } catch (err) {
      console.error('Demo error:', err);
    }
  }

  async loadDemoOffroadVideo() {
    this.sourceLabel.innerText = 'Loading off-road outdoor demo video...';
    try {
      const resp = await fetch('/api/load_demo_offroad', { method: 'POST' });
      const data = await resp.json();
      if (data.status === 'success') {
        this.currentVideoSource = 'Demo Off-Road Terrain';
        this.sourceLabel.innerText = `Demo: Off-Road Terrain (${data.metadata.total_frames} frames)`;
        this.map.clear();
        this.isPlaying = false;
        this.btnPlay.innerText = '▶ Play';
        this.smoothedDisplaySteer = 0.0;
        this.logConsole('[SOURCE] Loaded off-road outdoor terrain demo. Click Play to start.');
        this.stepForward();
      }
    } catch (err) {
      console.error('Offroad demo error:', err);
    }
  }

  async openWebcam() {
    this.sourceLabel.innerText = 'Opening USB webcam...';
    try {
      const resp = await fetch('/api/open_webcam', { method: 'POST' });
      const data = await resp.json();
      if (data.status === 'success') {
        this.currentVideoSource = 'Live Webcam';
        this.sourceLabel.innerText = 'Source: Live Webcam 0';
        this.map.clear();
        this.smoothedDisplaySteer = 0.0;
        this.logConsole('[SOURCE] Live Webcam connected. Click Play to stream.');
        this.togglePlay();
      }
    } catch (err) {
      console.error('Webcam error:', err);
    }
  }

  handleServerMessage(msg) {
    if (msg.type === 'FRAME_RESULT') {
      const data = msg.payload;
      const telem = data.telemetry;
      const vo = data.vo_pose;

      // 1. Update Video Images
      if (data.visualizations) {
        if (data.visualizations.perception_annotated) {
          this.feedMain.src = data.visualizations.perception_annotated;
        }
        if (data.visualizations.traversable_mask) {
          this.feedTraversable.src = data.visualizations.traversable_mask;
        }
        if (data.visualizations.optical_flow) {
          this.feedOpticalFlow.src = data.visualizations.optical_flow;
        }
      }

      // 2. Action Banner
      this.actionBanner.innerText = `ACTION: ${telem.action}`;
      this.actionBanner.className = 'action-banner';
      if (telem.alert_level === 'CAUTION') this.actionBanner.classList.add('caution');
      if (telem.alert_level === 'CRITICAL') this.actionBanner.classList.add('critical');

      // 3. Telemetry Numbers (whole integer values without decimal points)
      this.statSpeed.innerText = `${Math.round(telem.speed_kmh)} km/h`;
      this.statSteer.innerText = `${telem.steering_angle_deg >= 0 ? '+' : ''}${telem.steering_angle_deg.toFixed(1)}°`;
      this.statDist.innerText = `${Math.round(telem.total_distance_m)} m`;
      this.statObsCount.innerText = `${telem.obstacle_count}`;
      this.statNearestObs.innerText = telem.min_obstacle_dist < 90 ? `${telem.min_obstacle_dist.toFixed(1)} m` : '--';
      this.statVoPose.innerText = `(${vo.x.toFixed(1)}, ${vo.y.toFixed(1)})`;
      this.statFeatures.innerText = `${telem.features_tracked}`;

      // 4. Smoothed Steering Dial Needle (Decreased sensitivity, stable centering)
      const targetSteer = telem.steering_angle_deg;
      this.smoothedDisplaySteer = this.smoothedDisplaySteer * 0.70 + targetSteer * 0.30;
      if (Math.abs(this.smoothedDisplaySteer) < 0.8) {
        this.smoothedDisplaySteer = 0.0;
      }
      // Map [-30°, +30°] to [10%, 90%] for gentle, controlled movement
      const steerPct = Math.max(10, Math.min(90, 50 + (this.smoothedDisplaySteer / 30.0) * 40));
      this.steeringIndicator.style.left = `${steerPct}%`;

      // 5. Update Reconstructed 2D Trajectory Map
      this.map.addPoint(vo, telem.alert_level);

      // 6. Throttled Console Log (every 10 frames or on action change)
      if (data.frame % 10 === 0 || telem.action !== this.lastAction) {
        this.lastAction = telem.action;
        const line = `[#${String(data.frame).padStart(4, '0')}] VO:(${vo.x.toFixed(1)},${vo.y.toFixed(1)})m | Spd:${Math.round(telem.speed_kmh)}km/h | Steer:${telem.steering_angle_deg.toFixed(1)}° | Obs:${telem.obstacle_count} | ${telem.action}`;
        this.logConsole(line);
      }

    } else if (msg.type === 'VIDEO_ENDED') {
      this.isPlaying = false;
      this.btnPlay.innerText = '▶ Play';
      this.logConsole('🎬 [COMPLETE] End of video reached.');
      if (msg.payload && msg.payload.plot_image) {
        this.showPlotModal(msg.payload.plot_image);
      }

    } else if (msg.type === 'RUN_FINISHED') {
      if (msg.payload && msg.payload.plot_image) {
        this.showPlotModal(msg.payload.plot_image);
      }
    }
  }

  showPlotModal(imageSrc) {
    this.plotImg.src = imageSrc;
    this.plotModal.classList.add('active');
  }

  logConsole(text) {
    this.consoleOutput.innerText += text + '\n';
    this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.dashboard = new DashboardController();
});
