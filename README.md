# Advanced Camera Detection and Perception with Path Planner for UGV's

A real-time computer vision pipeline for autonomous navigation and driver assistance using **Dash Camera footage**.

The system consumes video files (MP4, AVI, MOV) or live camera feeds, executing **classical computer vision perception** (no deep learning required), **monocular visual odometry** with ground-plane perspective scaling for accurate speed & trajectory reconstruction, and **reactive path planning / steering advice**.

---

## System Capabilities

1. **Video Ingestion Modes**:
   - **File Upload**: Upload any dash camera footage (MP4, AVI, MOV) directly through the browser dashboard.
   - **CLI Pipeline**: Process videos offline with high-speed rendering:
     ```powershell
     python run_video_pipeline.py --video "path/to/footage.mp4" --show --save-video
     ```
   - **Live Webcam / USB Capture Card**: Connect a real camera device for live real-time analysis (`--webcam 0`).

2. **Classical CV Road Perception (`modules/perception.py`)**:
   - **Dynamic Road Color Sampling**: Dynamically learns the road surface color distribution in LAB/HSV color space, accommodating changing asphalt shades, gravel, and lighting shifts.
   - **CLAHE Lighting Equalization**: Counters harsh sunlight and tree shadows.
   - **Traversable Corridor & Hazard Mask**: Generates binary road masks (safe path = white, hazards/off-road = black).
   - **Obstacle Detection**: Detects contrasting vehicles, obstacles, and curbs with distance estimation (meters) and sector classification (Left, Center, Right).

3. **Monocular Visual Odometry (`modules/visual_odometry.py`)**:
   - **Feature Tracking**: Lucas-Kanade optical flow on Shi-Tomasi corners across road and roadside structures.
   - **Forward-Backward Cross Checking**: Rejects false tracks and moving dynamic objects.
   - **Ego-Motion Integration**: Computes frame-to-frame vehicle yaw rotation $\Delta \theta$ and forward translation $\Delta d$ to reconstruct the car's driving path $(x, y, \theta)$ without GPS.

4. **Steering Advisory & Driver Assistance (`modules/planner.py`)**:
   - Evaluates obstacle proximity and corridor clearance across 5 sectors.
   - Outputs:
     - **Action**: `MAINTAIN COURSE (PATH CLEAR)`, `STEER LEFT`, `STEER RIGHT`, or `CRITICAL (HAZARD AHEAD)`.
     - **Steering Angle**: In degrees ($-25^\circ$ to $+25^\circ$).
     - **Alert Level**: `NORMAL`, `CAUTION`, `CRITICAL`.

5. **Multi-Feed Dashboard & Trajectory Evaluation**:
   - Live annotated video feed (green road corridor, red/amber obstacle bboxes).
   - Binary traversability mask & optical flow vector HUD.
   - 2D Reconstructed Trajectory Map showing the car's path over time.
   - Steering dial gauge & telemetry stats (speed, distance, obstacle distance).
   - Automated export of **Matplotlib trajectory plot (`.png`)** and **CSV telemetry log**.

---

## Directory Structure

```
E:\SIH 2\
├── requirements.txt            # Python dependencies (OpenCV, FastAPI, NumPy, Matplotlib)
├── run_server.py               # Local FastAPI web server & WebSocket pipeline
├── run_server.bat              # One-click Windows batch launcher
├── run_video_pipeline.py       # Standalone CLI video processor
├── README.md                   # System documentation
├── modules\
│   ├── perception.py           # Road corridor segmentation & obstacle detector
│   ├── visual_odometry.py      # Monocular Lucas-Kanade ego-motion tracking
│   ├── planner.py              # Steering advisory & hazard avoidance logic
│   ├── video_processor.py      # Video decoding & demo video generator
│   ├── ugv_controller.py       # Closed-loop pipeline orchestrator
│   └── logger.py               # CSV logger & Matplotlib trajectory plotter
├── static\
│   ├── index.html              # Dashboard UI
│   ├── css\style.css           # Modern dark-theme styling
│   └── js\
│       ├── dashboard.js        # Video playback, upload handler & telemetry
│       └── trajectory_map.js   # Real-time 2D Canvas path renderer
└── tests\
    ├── test_perception.py      # Perception segmentation tests
    ├── test_visual_odometry.py # Visual odometry motion tracking tests
    └── test_planner.py         # Steering recommendation tests
```

---

## Sample Evaluation Videos

Pre-recorded driving datasets and dashcam footage for evaluation can be downloaded from Google Drive:

- 📦 **[Download Sample videos for uploading (Google Drive)](https://drive.google.com/drive/folders/1YD4zZ_SOs_tRH7RVWDh8FgeP4qsm6MA7?usp=sharing)**

**Included Scenarios:**
- `Sample 1.mp4`: Front dashcam highway/road driving with clear road markings.
- `Sample 2.mp4`: Offroad & rugged terrain navigation testing corridor adaptation.
- `Sample 3.mp4`: Complex multi-vehicle traffic and obstacle avoidance scenario.

Extract the downloaded videos/zip into the `uploads/` directory to quickly select and benchmark them in the web dashboard or CLI pipeline.

---

### 📲📲Method 1: Opening Interactive Web Dashboard (Recommended)📲📲

1. Launch the server:
   Double-click **`run_server.bat`** located in the downloaded zip file or run:
   ```powershell
   python run_server.py
   ```
2. Open `http://127.0.0.1:8000` in your web browser or it opens automatically.
3. Click **📁 Upload Video File** to select your real car video file, or click **🎬 Load Demo Video** to test with the built-in realistic clip.
4. Click **▶ Play** to start processing:
   - Watch the road corridor highlighted in green and obstacles boxed in red.
   - Watch the optical flow feature vectors tracking motion in real time.
   - Observe the 2D Trajectory Map reconstructing the car's path.
   - Watch the steering gauge indicate recommended maneuvers.
5. Click **📊 View Trajectory Plot** to inspect the trajectory plot and download `trajectory_log.csv`.

---

### Method 2: Command-Line Interface (CLI)

To process your car video file directly from the terminal and save an annotated video:
```powershell
python run_video_pipeline.py --video "path/to/my_car_video.mp4" --show --save-video
```

To test with the demo driving clip:
```powershell
python run_video_pipeline.py --demo --show --save-video
```

To run with a live USB camera:
```powershell
python run_video_pipeline.py --webcam 0 --show
```

---

### Method 3: Run Automated Unit Tests

```powershell
pytest tests/ -v
```
