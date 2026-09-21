/**
 * Real-Time Trajectory Map Renderer for Real Car Video
 * Dynamically auto-scales and renders the vehicle's 2D reconstructed driving path
 * from monocular Visual Odometry (GPS-denied).
 */

class TrajectoryMap {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');

    this.voPath = [];
    this.obstacleEvents = [];

    // Auto-scaling bounding box
    this.minX = -5.0;
    this.maxX = 5.0;
    this.minY = -2.0;
    this.maxY = 25.0;

    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  resizeCanvas() {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * (window.devicePixelRatio || 1);
    this.canvas.height = rect.height * (window.devicePixelRatio || 1);
    this.render();
  }

  clear() {
    this.voPath = [];
    this.obstacleEvents = [];
    this.minX = -5.0;
    this.maxX = 5.0;
    this.minY = -2.0;
    this.maxY = 25.0;
    this.render();
  }

  addPoint(voPoint, alertLevel = 'NORMAL') {
    this.voPath.push(voPoint);

    if (alertLevel === 'CRITICAL') {
      this.obstacleEvents.push(voPoint);
    }

    // Update dynamic bounding box
    const x = voPoint.x;
    const y = voPoint.y;
    this.minX = Math.min(this.minX, x - 4.0);
    this.maxX = Math.max(this.maxX, x + 4.0);
    this.minY = Math.min(this.minY, y - 2.0);
    this.maxY = Math.max(this.maxY, y + 8.0);

    this.render();
  }

  worldToCanvas(wx, wy) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const padding = 35;

    const usableW = w - padding * 2;
    const usableH = h - padding * 2;

    const rangeX = Math.max(10.0, this.maxX - this.minX);
    const rangeY = Math.max(15.0, this.maxY - this.minY);

    const scale = Math.min(usableW / rangeX, usableH / rangeY);

    const cx = padding + (wx - this.minX) * scale;
    // Invert Y so forward driving is pointing up
    const cy = h - (padding + (wy - this.minY) * scale);

    return { x: cx, y: cy, scale };
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // 1. Background
    ctx.fillStyle = '#080c14';
    ctx.fillRect(0, 0, w, h);

    // 2. Metric Grid
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#1a233a';
    ctx.font = '10px monospace';
    ctx.fillStyle = '#475569';

    const stepM = 10;
    const startGridX = Math.floor(this.minX / stepM) * stepM;
    const endGridX = Math.ceil(this.maxX / stepM) * stepM;
    for (let gx = startGridX; gx <= endGridX; gx += stepM) {
      const p1 = this.worldToCanvas(gx, this.minY);
      const p2 = this.worldToCanvas(gx, this.maxY);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
      ctx.fillText(`${gx}m`, p1.x + 2, h - 8);
    }

    const startGridY = Math.floor(this.minY / stepM) * stepM;
    const endGridY = Math.ceil(this.maxY / stepM) * stepM;
    for (let gy = startGridY; gy <= endGridY; gy += stepM) {
      const p1 = this.worldToCanvas(this.minX, gy);
      const p2 = this.worldToCanvas(this.maxX, gy);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
      ctx.fillText(`${gy}m`, 8, p1.y - 4);
    }

    // 3. Start Location Marker (Point (0, 0))
    const pStart = this.worldToCanvas(0.0, 0.0);
    ctx.beginPath();
    ctx.arc(pStart.x, pStart.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#10b981';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#86efac';
    ctx.font = 'bold 11px sans-serif';
    ctx.fillText('Start (0,0)', pStart.x + 10, pStart.y + 4);

    // 4. Draw Reconstructed Path (VO)
    if (this.voPath.length > 1) {
      ctx.beginPath();
      ctx.lineWidth = 3.0;
      ctx.strokeStyle = '#38bdf8';
      for (let i = 0; i < this.voPath.length; i++) {
        const pt = this.worldToCanvas(this.voPath[i].x, this.voPath[i].y);
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
    }

    // 5. Draw Obstacle Warning Event Markers (Red cross)
    for (const obs of this.obstacleEvents) {
      const pos = this.worldToCanvas(obs.x, obs.y);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      const s = 6;
      ctx.beginPath();
      ctx.moveTo(pos.x - s, pos.y - s);
      ctx.lineTo(pos.x + s, pos.y + s);
      ctx.moveTo(pos.x + s, pos.y - s);
      ctx.lineTo(pos.x - s, pos.y + s);
      ctx.stroke();
    }

    // 6. Current Vehicle Position and Heading Cone
    if (this.voPath.length > 0) {
      const cur = this.voPath[this.voPath.length - 1];
      const curPos = this.worldToCanvas(cur.x, cur.y);

      // Orientation arrow
      const arrowLen = 18;
      const targetX = curPos.x + Math.cos(cur.theta) * arrowLen;
      const targetY = curPos.y - Math.sin(cur.theta) * arrowLen;

      ctx.beginPath();
      ctx.moveTo(curPos.x, curPos.y);
      ctx.lineTo(targetX, targetY);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 3;
      ctx.stroke();

      // Vehicle dot
      ctx.beginPath();
      ctx.arc(curPos.x, curPos.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#0284c7';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}
