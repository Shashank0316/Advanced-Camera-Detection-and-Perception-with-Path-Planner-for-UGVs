/**
 * UGV 3D Simulation Engine (Three.js)
 * Simulates the outdoor environment, obstacles, UGV differential kinematics,
 * and virtual onboard camera streaming.
 */

class UGVWorldSimulation {
  constructor(containerId, offscreenResolution = { width: 360, height: 270 }) {
    this.container = document.getElementById(containerId);
    this.offscreenRes = offscreenResolution;

    // Kinematic State (Ground Truth)
    this.gtPose = { x: 0.0, y: 0.0, theta: 0.0 }; // x, y on ground, theta is yaw
    this.linearVelocity = 0.0;
    this.angularVelocity = 0.0;

    // Start and Goal Coordinates
    this.startPose = { x: 0.0, y: 0.0 };
    this.goalPose = { x: 22.0, y: 14.0 };
    this.goalTolerance = 1.2;

    // Default Static Obstacles (Course layout)
    this.obstacles = [
      { x: 5.5, y: 2.5, radius: 1.1, type: "rock" },
      { x: 10.0, y: 6.8, radius: 1.2, type: "crate" },
      { x: 13.5, y: 5.2, radius: 1.0, type: "rock" },
      { x: 16.0, y: 11.2, radius: 1.2, type: "crate" },
      { x: 18.5, y: 9.5, radius: 1.1, type: "rock" },
    ];
    this.obstacleMeshes = [];

    // Camera modes: 'chase', 'topdown', 'fpv'
    this.currentCameraView = 'chase';

    // Init Three.js
    this.initScene();
    this.buildEnvironment();
    this.buildUGV();
    this.initOffscreenCamera();

    // Event listener for resize
    window.addEventListener('resize', () => this.onWindowResize());
  }

  initScene() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x87ceeb); // Sky blue
    this.scene.fog = new THREE.FogExp2(0x87ceeb, 0.015);

    // Main Viewport Camera
    this.mainCamera = new THREE.PerspectiveCamera(60, width / height, 0.1, 200);
    this.mainCamera.position.set(-6, 8, -6);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setSize(width, height);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // Lights
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.7);
    hemiLight.position.set(0, 50, 0);
    this.scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xfffaed, 1.1);
    dirLight.position.set(30, 45, 20);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 100;
    const d = 30;
    dirLight.shadow.camera.left = -d;
    dirLight.shadow.camera.right = d;
    dirLight.shadow.camera.top = d;
    dirLight.shadow.camera.bottom = -d;
    this.scene.add(dirLight);
  }

  createProceduralGroundTexture() {
    // Generate an earthy outdoor soil/grass canvas texture
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    // Base earth tone (Olive/khaki green-brown)
    ctx.fillStyle = '#6b705c';
    ctx.fillRect(0, 0, 512, 512);

    // Add noise and speckles for realistic outdoor soil
    for (let i = 0; i < 15000; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      const r = Math.random() * 2.5;
      ctx.fillStyle = Math.random() > 0.5 ? '#585e4a' : '#7f876f';
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Dirt trails
    for (let i = 0; i < 500; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      ctx.fillStyle = '#a5a58d';
      ctx.fillRect(x, y, Math.random() * 3, Math.random() * 3);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(25, 25);
    return texture;
  }

  buildEnvironment() {
    // Ground Plane (X: World X, Z: World Y, Y: Height/Up)
    const groundGeo = new THREE.PlaneGeometry(120, 120);
    const groundTex = this.createProceduralGroundTexture();
    const groundMat = new THREE.MeshLambertMaterial({ map: groundTex });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // Course boundary markers (slight posts)
    this.buildCourseBoundaries();

    // Start Point A (Green ring pad)
    const startGeo = new THREE.RingGeometry(0.2, 1.4, 32);
    const startMat = new THREE.MeshBasicMaterial({ color: 0x22c55e, side: THREE.DoubleSide });
    const startMesh = new THREE.Mesh(startGeo, startMat);
    startMesh.rotation.x = -Math.PI / 2;
    startMesh.position.set(this.startPose.x, 0.02, this.startPose.y);
    this.scene.add(startMesh);

    // Goal Point B (Red glowing beacon pedestal)
    const goalPedestalGeo = new THREE.CylinderGeometry(1.2, 1.3, 0.2, 32);
    const goalPedestalMat = new THREE.MeshLambertMaterial({ color: 0xef4444 });
    const goalPedestal = new THREE.Mesh(goalPedestalGeo, goalPedestalMat);
    goalPedestal.position.set(this.goalPose.x, 0.1, this.goalPose.y);
    this.scene.add(goalPedestal);

    // Glowing beacon core
    const beaconGeo = new THREE.OctahedronGeometry(0.6, 1);
    const beaconMat = new THREE.MeshStandardMaterial({
      color: 0xff3333,
      emissive: 0xff2222,
      emissiveIntensity: 0.8,
      roughness: 0.2,
    });
    this.beaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
    this.beaconMesh.position.set(this.goalPose.x, 1.5, this.goalPose.y);
    this.scene.add(this.beaconMesh);

    // Beacon Light
    const beaconLight = new THREE.PointLight(0xff3333, 2.0, 15);
    beaconLight.position.set(this.goalPose.x, 2.0, this.goalPose.y);
    this.scene.add(beaconLight);

    // Build Static Obstacles
    this.spawnObstacles();
  }

  buildCourseBoundaries() {
    // Boundary posts around perimeter
    const postGeo = new THREE.CylinderGeometry(0.12, 0.12, 1.2, 12);
    const postMat = new THREE.MeshLambertMaterial({ color: 0xf59e0b });
    const xMin = -5, xMax = 28, yMin = -6, yMax = 20;

    for (let x = xMin; x <= xMax; x += 4) {
      this.createPost(postGeo, postMat, x, yMin);
      this.createPost(postGeo, postMat, x, yMax);
    }
    for (let y = yMin; y <= yMax; y += 4) {
      this.createPost(postGeo, postMat, xMin, y);
      this.createPost(postGeo, postMat, xMax, y);
    }
  }

  createPost(geo, mat, x, z) {
    const post = new THREE.Mesh(geo, mat);
    post.position.set(x, 0.6, z);
    post.castShadow = true;
    this.scene.add(post);
  }

  spawnObstacles() {
    // Clear previous meshes
    for (const mesh of this.obstacleMeshes) {
      this.scene.remove(mesh);
    }
    this.obstacleMeshes = [];

    // Rock Material (Dark Slate Gray / Granite)
    const rockMat = new THREE.MeshStandardMaterial({
      color: 0x3d4451,
      roughness: 0.9,
      metalness: 0.1,
    });

    // Crate Material (Reddish Cedar Wood)
    const crateMat = new THREE.MeshStandardMaterial({
      color: 0x8b5a2b,
      roughness: 0.7,
      metalness: 0.05,
    });

    for (const obs of this.obstacles) {
      let mesh;
      if (obs.type === "rock") {
        // Irregular rock boulder
        const geo = new THREE.DodecahedronGeometry(obs.radius, 1);
        mesh = new THREE.Mesh(geo, rockMat);
        mesh.position.set(obs.x, obs.radius * 0.75, obs.y);
        mesh.scale.set(1.0, 0.85 + Math.random() * 0.3, 1.0 + Math.random() * 0.2);
      } else {
        // Cargo crate / barricade
        const size = obs.radius * 1.6;
        const geo = new THREE.BoxGeometry(size, size * 0.8, size);
        mesh = new THREE.Mesh(geo, crateMat);
        mesh.position.set(obs.x, (size * 0.8) / 2, obs.y);
        mesh.rotation.y = Math.random() * Math.PI;
      }

      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.scene.add(mesh);
      this.obstacleMeshes.push(mesh);
    }
  }

  randomizeObstacles() {
    this.obstacles = [
      { x: 4.5 + Math.random() * 2, y: 1.5 + Math.random() * 2, radius: 1.0 + Math.random() * 0.3, type: "rock" },
      { x: 9.0 + Math.random() * 3, y: 5.0 + Math.random() * 3, radius: 1.1 + Math.random() * 0.3, type: "crate" },
      { x: 13.0 + Math.random() * 2, y: 4.0 + Math.random() * 3, radius: 1.0 + Math.random() * 0.3, type: "rock" },
      { x: 15.5 + Math.random() * 3, y: 9.5 + Math.random() * 3, radius: 1.2 + Math.random() * 0.3, type: "crate" },
      { x: 18.0 + Math.random() * 2, y: 8.5 + Math.random() * 3, radius: 1.1 + Math.random() * 0.3, type: "rock" },
    ];
    this.spawnObstacles();
  }

  buildUGV() {
    this.ugvGroup = new THREE.Group();

    // 1. Chassis Body (Sturdy UGV in Industrial Safety Orange)
    const bodyGeo = new THREE.BoxGeometry(1.2, 0.45, 0.8);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0xf97316,
      roughness: 0.35,
      metalness: 0.3,
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.4;
    body.castShadow = true;
    this.ugvGroup.add(body);

    // Chassis top plate / protective cage (Dark metal)
    const plateGeo = new THREE.BoxGeometry(0.85, 0.15, 0.65);
    const plateMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.6 });
    const plate = new THREE.Mesh(plateGeo, plateMat);
    plate.position.set(-0.05, 0.65, 0);
    plate.castShadow = true;
    this.ugvGroup.add(plate);

    // 2. Wheels (4 Rugged Black Rubber Tires)
    const wheelGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.18, 20);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.9 });
    this.wheels = [];

    const wheelOffsets = [
      { x: 0.45, z: 0.48 },   // Front-Left
      { x: 0.45, z: -0.48 },  // Front-Right
      { x: -0.45, z: 0.48 },  // Rear-Left
      { x: -0.45, z: -0.48 }, // Rear-Right
    ];

    for (const offset of wheelOffsets) {
      const wheel = new THREE.Mesh(wheelGeo, wheelMat);
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(offset.x, 0.24, offset.z);
      wheel.castShadow = true;
      this.ugvGroup.add(wheel);
      this.wheels.push(wheel);
    }

    // 3. Sensor Mast & Onboard Camera Housing (Front)
    const mastGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.3, 12);
    const mastMat = new THREE.MeshStandardMaterial({ color: 0x334155 });
    const mast = new THREE.Mesh(mastGeo, mastMat);
    mast.position.set(0.5, 0.65, 0);
    this.ugvGroup.add(mast);

    // Camera Lens Box
    const camBoxGeo = new THREE.BoxGeometry(0.18, 0.14, 0.2);
    const camBoxMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.2 });
    const camBox = new THREE.Mesh(camBoxGeo, camBoxMat);
    camBox.position.set(0.55, 0.8, 0);
    this.ugvGroup.add(camBox);

    // Blue Camera Lens Glass
    const lensGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.05, 16);
    const lensMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });
    const lens = new THREE.Mesh(lensGeo, lensMat);
    lens.rotation.z = Math.PI / 2;
    lens.position.set(0.65, 0.8, 0);
    this.ugvGroup.add(lens);

    // Headlights (Twin white beams)
    const lightL = new THREE.SpotLight(0xffffff, 2.5, 18, Math.PI / 5, 0.4);
    lightL.position.set(0.65, 0.45, 0.25);
    lightL.target.position.set(5.0, 0.0, 0.25);
    this.ugvGroup.add(lightL);
    this.ugvGroup.add(lightL.target);

    const lightR = new THREE.SpotLight(0xffffff, 2.5, 18, Math.PI / 5, 0.4);
    lightR.position.set(0.65, 0.45, -0.25);
    lightR.target.position.set(5.0, 0.0, -0.25);
    this.ugvGroup.add(lightR);
    this.ugvGroup.add(lightR.target);

    this.scene.add(this.ugvGroup);
    this.syncUGVTransform();
  }

  initOffscreenCamera() {
    // Virtual Onboard Navigation Camera mounted on UGV
    const aspect = this.offscreenRes.width / this.offscreenRes.height;
    this.onboardCamera = new THREE.PerspectiveCamera(72, aspect, 0.1, 40);

    // Mount inside UGV group facing forward (+X direction in UGV local frame), slightly pitched down
    this.onboardCamera.position.set(0.65, 0.8, 0);
    // Three.js cameras default look along -Z; rotate so it looks along +X with a -12 deg pitch down
    this.onboardCamera.rotation.order = 'YXZ';
    this.onboardCamera.rotation.y = -Math.PI / 2;
    this.onboardCamera.rotation.x = -0.18; // ~10.5 degree downward pitch

    this.ugvGroup.add(this.onboardCamera);

    // Offscreen Canvas and Dedicated Renderer for onboard feed
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = this.offscreenRes.width;
    this.offscreenCanvas.height = this.offscreenRes.height;

    this.offscreenRenderer = new THREE.WebGLRenderer({
      canvas: this.offscreenCanvas,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    this.offscreenRenderer.setSize(this.offscreenRes.width, this.offscreenRes.height);
  }

  captureOnboardFrameBase64() {
    // Render the onboard camera view to the offscreen canvas
    this.offscreenRenderer.render(this.scene, this.onboardCamera);
    return this.offscreenCanvas.toDataURL('image/jpeg', 0.82);
  }

  syncUGVTransform() {
    // Ground Truth: world X -> Three.js X, world Y -> Three.js Z
    this.ugvGroup.position.x = this.gtPose.x;
    this.ugvGroup.position.z = this.gtPose.y;
    // Yaw theta around Three.js vertical Y axis
    this.ugvGroup.rotation.y = -this.gtPose.theta;
  }

  updatePhysics(dt = 0.05) {
    // Integrate differential drive kinematics (Ground Truth update)
    const v = this.linearVelocity;
    const w = this.angularVelocity;

    this.gtPose.theta += w * dt;
    // Normalize theta to [-PI, PI]
    this.gtPose.theta = ((this.gtPose.theta + Math.PI) % (2 * Math.PI)) - Math.PI;

    this.gtPose.x += v * Math.cos(this.gtPose.theta) * dt;
    this.gtPose.y += v * Math.sin(this.gtPose.theta) * dt;

    this.syncUGVTransform();

    // Spin wheels according to linear velocity
    const wheelRot = (v * dt) / 0.24;
    for (const wMesh of this.wheels) {
      wMesh.rotation.y += wheelRot;
    }

    // Pulse beacon animation
    if (this.beaconMesh) {
      this.beaconMesh.rotation.y += 0.03;
      this.beaconMesh.position.y = 1.5 + Math.sin(Date.now() * 0.004) * 0.15;
    }
  }

  setMotorCommands(v, w) {
    this.linearVelocity = v;
    this.angularVelocity = w;
  }

  reset(startX = 0.0, startY = 0.0, startTheta = 0.0) {
    this.gtPose.x = startX;
    this.gtPose.y = startY;
    this.gtPose.theta = startTheta;
    this.linearVelocity = 0.0;
    this.angularVelocity = 0.0;
    this.syncUGVTransform();
  }

  setCameraView(mode) {
    this.currentCameraView = mode;
  }

  render() {
    // Update main viewport camera position based on current view mode
    const px = this.gtPose.x;
    const pz = this.gtPose.y;
    const th = this.gtPose.theta;

    if (this.currentCameraView === 'chase') {
      // Smooth chase camera following behind vehicle
      const dist = 4.2;
      const height = 2.4;
      const targetCamX = px - Math.cos(th) * dist;
      const targetCamZ = pz - Math.sin(th) * dist;

      this.mainCamera.position.x += (targetCamX - this.mainCamera.position.x) * 0.12;
      this.mainCamera.position.y += (height - this.mainCamera.position.y) * 0.12;
      this.mainCamera.position.z += (targetCamZ - this.mainCamera.position.z) * 0.12;

      this.mainCamera.lookAt(px + Math.cos(th) * 2.0, 0.5, pz + Math.sin(th) * 2.0);

    } else if (this.currentCameraView === 'topdown') {
      // Overhead tactical view
      this.mainCamera.position.set(11, 24, 7);
      this.mainCamera.lookAt(11, 0, 7);

    } else if (this.currentCameraView === 'fpv') {
      // Direct driver's eye view
      this.mainCamera.position.set(px + Math.cos(th) * 0.6, 0.85, pz + Math.sin(th) * 0.6);
      this.mainCamera.lookAt(px + Math.cos(th) * 6, 0.4, pz + Math.sin(th) * 6);
    }

    this.renderer.render(this.scene, this.mainCamera);
  }

  onWindowResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.mainCamera.aspect = width / height;
    this.mainCamera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
}
