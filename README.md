# VisionNav

### Vision-Based Autonomous Navigation for Unmanned Ground Vehicles

**Smart India Hackathon 2026 — SIH26126**  
**Organization:** Bharat Electronics Limited  
**Theme:** Smart Automation  
**Team:** Philos — GITAM081

VisionNav is a CPU-only computer vision pipeline for autonomous navigation of unmanned ground vehicles in outdoor environments.

The system uses a single camera to detect obstacles, estimate vehicle motion without GPS, and generate steering decisions through a modular perception, visual odometry and path-planning pipeline.

## Problem

Outdoor UGVs operate in environments where:

- GPS may be unavailable or unreliable
- Lighting conditions can change significantly
- Painted road markings can be incorrectly detected as obstacles
- Paved-to-dirt transitions can confuse fixed visual thresholds
- A monocular camera does not directly provide metric depth
- Heavy deep-learning models can be difficult to deploy on constrained edge hardware

## Proposed Solution

VisionNav addresses these challenges using a classical, explainable computer-vision pipeline that runs on standard CPU hardware.

```text
Camera / Dashcam Video
          ↓
Frame Capture & Preprocessing
          ↓
Perception
          ↓
Obstacle Detection
          ↓
Visual Odometry
          ↓
Pose + Speed Estimation
          ↓
Path Planner
          ↓
Steering / Emergency Brake
          ↓
Live Web Dashboard