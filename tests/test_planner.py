"""
Unit tests for Driver Assistance & Path Planning Module
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.planner import PathPlanner


def test_planner_clear_road():
    planner = PathPlanner()
    current_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
    densities = {"far_left": 0.0, "left": 0.0, "center": 0.0, "right": 0.0, "far_right": 0.0}
    obstacles = []

    plan = planner.plan_step(current_pose, densities, obstacles)

    assert plan["alert_level"] == "NORMAL"
    assert "MAINTAIN COURSE" in plan["recommended_action"]
    assert abs(plan["steering_angle_deg"]) <= 5.0


def test_planner_obstacle_ahead_steer_left():
    planner = PathPlanner()
    current_pose = {"x": 0.0, "y": 10.0, "theta": 0.0}
    densities = {"far_left": 0.0, "left": 0.0, "center": 0.35, "right": 0.40, "far_right": 0.20}
    obstacles = [{"distance_est": 12.0, "sector": "Center"}]

    plan = planner.plan_step(current_pose, densities, obstacles)

    assert plan["alert_level"] in ["CAUTION", "CRITICAL"]
    # Right/Center obstructed, left is clear -> steer left (positive degrees)
    assert plan["steering_angle_deg"] > 0.0


def test_planner_critical_braking_zone():
    planner = PathPlanner(critical_braking_distance=8.0)
    current_pose = {"x": 0.0, "y": 15.0, "theta": 0.0}
    densities = {"far_left": 0.0, "left": 0.0, "center": 0.50, "right": 0.20, "far_right": 0.0}
    obstacles = [{"distance_est": 5.0, "sector": "Center"}]

    plan = planner.plan_step(current_pose, densities, obstacles)

    assert plan["alert_level"] == "CRITICAL"
    assert "HAZARD AHEAD" in plan["recommended_action"]
