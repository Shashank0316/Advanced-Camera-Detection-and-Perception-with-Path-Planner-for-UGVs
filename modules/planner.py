"""
Path Planning & Driver Assistance Module for Real Car Video (Smoothed & Stable)
Analyzes detected road obstacles and sector clearances to output real-time driving advisories,
recommended steering angles (degrees), and collision warning alerts.
Features low-pass smoothing and deadband filtering to prevent jittery/overly-sensitive steering.
"""

from typing import Dict, List, Any
import numpy as np


class PathPlanner:
    def __init__(
        self,
        critical_braking_distance: float = 6.5,
        caution_distance: float = 14.0,
        max_steering_deg: float = 22.0,
        obstacle_density_threshold: float = 0.16,
        smoothing_factor: float = 0.22,
        deadband_deg: float = 1.5,
    ):
        """
        Args:
            critical_braking_distance: Distance (m) below which emergency stop / slow down is triggered.
            caution_distance: Distance (m) to begin proactive avoidance steering.
            max_steering_deg: Maximum steering recommendation in degrees.
            obstacle_density_threshold: Density above which a sector is deemed blocked.
            smoothing_factor: EMA alpha for steering angle (lower = smoother, less sensitive).
            deadband_deg: Angles below this threshold are locked to 0° to prevent fidgeting.
        """
        self.critical_dist = critical_braking_distance
        self.caution_dist = caution_distance
        self.max_steer = max_steering_deg
        self.density_thresh = obstacle_density_threshold
        self.alpha = smoothing_factor
        self.deadband = deadband_deg

        # Filter state
        self.current_steering_deg: float = 0.0

    def reset(self):
        self.current_steering_deg = 0.0

    def plan_step(
        self,
        current_pose: Dict[str, float],
        sector_densities: Dict[str, float],
        detected_obstacles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate road conditions and compute smoothed steering recommendations.
        """
        d_far_left = sector_densities.get("far_left", 0.0)
        d_left = sector_densities.get("left", 0.0)
        d_center = sector_densities.get("center", 0.0)
        d_right = sector_densities.get("right", 0.0)
        d_far_right = sector_densities.get("far_right", 0.0)

        # Identify closest obstacle in critical driving path
        min_obstacle_dist = 999.0
        closest_sector = "None"

        for obs in detected_obstacles:
            dist = obs.get("distance_est", 999.0)
            if dist < min_obstacle_dist:
                min_obstacle_dist = dist
                closest_sector = obs.get("sector", "Center")

        raw_steering = 0.0
        action = "MAINTAIN COURSE (PATH CLEAR)"
        advisory = "Road corridor clear. Safe driving conditions."
        alert_level = "NORMAL"
        linear_v = 0.85

        # 1. Critical Hazard Check (Very close obstacle directly in path)
        if min_obstacle_dist <= self.critical_dist or (d_center > self.density_thresh * 2.2):
            left_clearance = 1.0 - (d_left * 0.6 + d_far_left * 0.4)
            right_clearance = 1.0 - (d_right * 0.6 + d_far_right * 0.4)

            if left_clearance >= right_clearance:
                raw_steering = min(self.max_steer, 14.0)
                action = "STEER LEFT (HAZARD AHEAD)"
            else:
                raw_steering = -min(self.max_steer, 14.0)
                action = "STEER RIGHT (HAZARD AHEAD)"

            advisory = f"CRITICAL: Obstacle {min_obstacle_dist:.1f}m ahead! Slow down."
            alert_level = "CRITICAL"
            linear_v = 0.25

        # 2. Caution Zone (Obstacle approaching within caution distance)
        elif min_obstacle_dist <= self.caution_dist or (d_center > self.density_thresh):
            left_clearance = 1.0 - (d_left * 0.7 + d_far_left * 0.3)
            right_clearance = 1.0 - (d_right * 0.7 + d_far_right * 0.3)

            # Gradual, gentle steering adjustment
            proximity_factor = (self.caution_dist - min_obstacle_dist) / max(1.0, self.caution_dist)

            if closest_sector in ["Center", "Right", "Far-Right"] and left_clearance >= right_clearance:
                raw_steering = 5.0 + proximity_factor * 8.0
                action = "STEER LEFT (CLEAR CORRIDOR)"
            elif closest_sector in ["Center", "Left", "Far-Left"]:
                raw_steering = -(5.0 + proximity_factor * 8.0)
                action = "STEER RIGHT (CLEAR CORRIDOR)"
            else:
                raw_steering = 0.0
                action = "SLOW DOWN"

            advisory = f"CAUTION: Obstacle in {closest_sector} at {min_obstacle_dist:.1f}m."
            alert_level = "CAUTION"
            linear_v = 0.55

        # 3. Path Clear: Gentle corridor centering with high tolerance
        else:
            lane_imbalance = (d_left - d_right)
            # Gentle centering with reduced gain
            raw_steering = float(np.clip(lane_imbalance * 4.0, -3.0, 3.0))

        # --- Low-Pass Exponential Moving Average (EMA) Smoothing ---
        # Smooths out frame-to-frame noise and prevents rapid twitching
        self.current_steering_deg = (
            (1.0 - self.alpha) * self.current_steering_deg + self.alpha * raw_steering
        )

        # Apply deadband: micro-movements < deadband threshold are zeroed out during normal cruising
        output_steering = self.current_steering_deg
        if alert_level == "NORMAL" and abs(output_steering) < self.deadband:
            output_steering = 0.0

        output_steering = float(np.clip(output_steering, -self.max_steer, self.max_steer))

        return {
            "steering_angle_deg": round(output_steering, 1),
            "linear_velocity": linear_v,
            "recommended_action": action,
            "advisory_message": advisory,
            "alert_level": alert_level,
            "min_obstacle_dist": round(float(min_obstacle_dist), 1) if min_obstacle_dist < 100 else 99.0,
        }
