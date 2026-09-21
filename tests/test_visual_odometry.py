"""
Unit tests for Visual Odometry on Car Camera Footage
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.visual_odometry import VisualOdometry


def create_textured_road_frame() -> np.ndarray:
    np.random.seed(42)
    img = np.full((360, 640, 3), 110, dtype=np.uint8)

    # Sprinkle road texture and edge features
    for _ in range(350):
        x = np.random.randint(40, 600)
        y = np.random.randint(140, 340)
        color = (int(np.random.randint(0, 40)), int(np.random.randint(0, 40)), int(np.random.randint(0, 40)))
        cv2.circle(img, (x, y), np.random.randint(2, 5), color, -1)

    return img


def test_visual_odometry_initialization():
    vo = VisualOdometry()
    frame = create_textured_road_frame()
    res = vo.process_frame(frame)

    assert "pose" in res
    assert res["pose"]["x"] == 0.0
    assert res["pose"]["y"] == 0.0
    assert res["speed_kmh"] == 0.0
    assert res["tracked_features_count"] > 10


def test_visual_odometry_forward_motion():
    vo = VisualOdometry()
    frame1 = create_textured_road_frame()
    vo.process_frame(frame1)

    # Shift downward by 8 pixels to simulate forward motion
    M = np.float32([[1, 0, 0], [0, 1, 8]])
    frame2 = cv2.warpAffine(frame1, M, (640, 360))

    res2 = vo.process_frame(frame2, dt=0.033)

    assert res2["speed_kmh"] >= 0.0
    assert res2["total_distance"] >= 0.0
    assert res2["pose"]["y"] >= 0.0
