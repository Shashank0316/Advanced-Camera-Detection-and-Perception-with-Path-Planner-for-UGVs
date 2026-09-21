"""
Unit tests for Classical CV Perception Module on Car Camera Footage
Verifies:
1. Small white dashed lines and safety markings are completely excluded from obstacles.
2. Off-road terrains (dirt, gravel, grass) are classified as traversable.
3. Real 3D obstacles (boulders, vehicles, crates) are accurately detected.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.perception import PerceptionDetector


def create_road_frame_with_small_white_lines() -> np.ndarray:
    """Create a paved road frame with small white dashed lines, stripes, and edge lines."""
    h, w = 360, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Sky
    horizon_y = int(h * 0.40)
    frame[:horizon_y, :] = [230, 200, 140]

    # 2. Roadside
    frame[horizon_y:, :] = [45, 85, 52]

    # 3. Asphalt Road
    road_pts = np.array([
        [w // 2 - 40, horizon_y],
        [w // 2 + 40, horizon_y],
        [w // 2 + 220, h],
        [w // 2 - 220, h],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [road_pts], (75, 78, 82))

    # 4. Small White Dashed Lines & Side Strips (High brightness, low saturation)
    for y in range(horizon_y + 10, h, 28):
        # Small dashes: width 6px, height 14px
        cv2.line(frame, (w // 2, y), (w // 2, min(h - 1, y + 14)), (245, 245, 250), 6)

    # Continuous side safety lines
    cv2.line(frame, (w // 2 - 38, horizon_y), (w // 2 - 210, h), (240, 240, 245), 4)
    cv2.line(frame, (w // 2 + 38, horizon_y), (w // 2 + 210, h), (240, 240, 245), 4)

    return frame


def create_offroad_frame(has_boulder: bool = False) -> np.ndarray:
    """Create an off-road terrain frame (dirt, gravel, grass) with optional 3D boulder."""
    h, w = 360, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Sky
    horizon_y = int(h * 0.40)
    frame[:horizon_y, :] = [230, 200, 140]

    # 2. Off-road Dirt & Grass Terrain (Earthy brown/olive tones)
    frame[horizon_y:, :] = [40, 80, 50]  # Grass base

    # Dirt track corridor
    track_pts = np.array([
        [w // 2 - 50, horizon_y],
        [w // 2 + 50, horizon_y],
        [w // 2 + 230, h],
        [w // 2 - 230, h],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [track_pts], (55, 95, 125))  # Earthy brown dirt (BGR: 55, 95, 125)

    # Add flat small white stones / chalk streaks (should NOT be detected as obstacles)
    cv2.ellipse(frame, (w // 2 - 30, int(h * 0.7)), (12, 5), 0, 0, 360, (220, 225, 230), -1)
    cv2.ellipse(frame, (w // 2 + 40, int(h * 0.8)), (15, 6), 0, 0, 360, (215, 220, 225), -1)

    if has_boulder:
        # Prominent 3D Boulder (dark slate/rock with solid volumetric area and shadow)
        bx, by = int(w * 0.52), int(h * 0.62)
        bw, bh = 70, 55
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (35, 40, 45), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (15, 20, 25), 2)
        # Ground shadow under boulder
        cv2.rectangle(frame, (bx, by + bh - 5), (bx + bw + 10, by + bh + 8), (10, 10, 15), -1)

    return frame


def test_small_white_lines_are_not_detected_as_obstacles():
    """Verify that small white lines and dashes on paved roads are completely excluded."""
    detector = PerceptionDetector()
    frame = create_road_frame_with_small_white_lines()
    res = detector.process_frame(frame)

    assert len(res["obstacles"]) == 0
    assert res["sector_densities"]["center"] < 0.10


def test_offroad_terrain_traversability():
    """Verify that off-road terrain (dirt, grass, flat light stones) is classified as safe."""
    detector = PerceptionDetector()
    frame = create_offroad_frame(has_boulder=False)
    res = detector.process_frame(frame)

    # Flat dirt and light pebbles must NOT trigger obstacles
    assert len(res["obstacles"]) == 0
    assert res["sector_densities"]["center"] < 0.10


def test_offroad_boulder_detected():
    """Verify that genuine 3D obstacles (like boulders on off-road terrain) are detected."""
    detector = PerceptionDetector()
    frame = create_offroad_frame(has_boulder=True)
    # Stream 2 consecutive frames to confirm persistent obstacle track
    detector.process_frame(frame)
    res = detector.process_frame(frame)

    assert len(res["obstacles"]) == 1
    obs = res["obstacles"][0]
    assert obs["distance_est"] > 0.0
    assert obs["area"] >= 400


def create_road_frame_with_car(cx_offset: int = 0, cy_offset: int = 0, scale: float = 1.0) -> np.ndarray:
    """Create a paved road frame with a realistic vehicle in front."""
    frame = create_road_frame_with_small_white_lines()
    h, w = frame.shape[:2]

    # Base car dimensions
    bw = int(64 * scale)
    bh = int(46 * scale)
    bx = (w // 2 - bw // 2) + cx_offset
    by = 195 + cy_offset

    # 1. Undercarriage dark shadow (V < 52)
    cv2.rectangle(frame, (bx + 3, by + bh - 6), (bx + bw - 3, by + bh + 4), (15, 15, 18), -1)

    # 2. Vehicle Main Body (Dark Slate Blue BGR: 160, 50, 40)
    cv2.rectangle(frame, (bx, by + 12), (bx + bw, by + bh), (160, 50, 40), -1)
    # Roof / Cabin
    cabin_pts = np.array([
        [bx + int(bw * 0.18), by + 12],
        [bx + int(bw * 0.28), by],
        [bx + int(bw * 0.72), by],
        [bx + int(bw * 0.82), by + 12],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [cabin_pts], (140, 45, 35))

    # 3. Rear Windshield (Dark tinted glass)
    windshield_pts = np.array([
        [bx + int(bw * 0.22), by + 11],
        [bx + int(bw * 0.30), by + 2],
        [bx + int(bw * 0.70), by + 2],
        [bx + int(bw * 0.78), by + 11],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [windshield_pts], (45, 45, 50))

    # 4. Red Tail Lights (Bright red in BGR: 0, 0, 240)
    tl_w = max(4, int(bw * 0.18))
    tl_h = max(3, int(bh * 0.16))
    cv2.rectangle(frame, (bx + 4, by + 16), (bx + 4 + tl_w, by + 16 + tl_h), (0, 0, 240), -1)
    cv2.rectangle(frame, (bx + bw - 4 - tl_w, by + 16), (bx + bw - 4, by + 16 + tl_h), (0, 0, 240), -1)

    return frame


def test_moving_car_detected():
    """Verify that a moving car in front is detected with motion differencing and car signature."""
    detector = PerceptionDetector()

    # Frame 1: Car at initial position
    frame1 = create_road_frame_with_car(cx_offset=0, cy_offset=0, scale=1.0)
    res1 = detector.process_frame(frame1)

    # Frame 2: Car has moved forward/closer with scale change (creates inter-frame motion diff)
    frame2 = create_road_frame_with_car(cx_offset=6, cy_offset=8, scale=1.06)
    res2 = detector.process_frame(frame2)

    # Verify exactly 1 unified obstacle box is detected (no separate split boxes for the vehicle)
    assert len(res2["obstacles"]) == 1, f"Expected 1 unified obstacle box, got {len(res2['obstacles'])}"
    car_obs = res2["obstacles"][0]

    # Check that car is recognized and tracked as an Obstacle
    assert car_obs["label"] == "Obstacle"
    assert car_obs["distance_est"] > 0.0
    assert car_obs["area"] >= 200

