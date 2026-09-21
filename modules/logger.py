"""
Trajectory Logger & Analysis Module for Real Car Video
Logs frame-by-frame visual odometry poses, steering recommendations,
and generates multi-metric trajectory plots.
"""

import csv
import io
import os
from typing import Dict, List, Optional, Tuple, Any
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrajectoryLogger:
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_entries: List[Dict[str, Any]] = []
        self.is_active: bool = False

    def start_run(self):
        """Reset and start a fresh logging session."""
        self.log_entries = []
        self.is_active = True

    def log_step(
        self,
        frame_idx: int,
        time_sec: float,
        vo_pose: Dict[str, float],
        speed_kmh: float,
        steering_deg: float,
        action: str,
        alert_level: str,
        min_obs_dist: float,
        features_count: int,
    ):
        """Record telemetry for a single video frame."""
        if not self.is_active:
            return

        entry = {
            "frame": frame_idx,
            "time_sec": round(time_sec, 2),
            "vo_x": round(vo_pose.get("x", 0.0), 2),
            "vo_y": round(vo_pose.get("y", 0.0), 2),
            "vo_theta_deg": round(vo_pose.get("theta_deg", 0.0), 2),
            "speed_kmh": int(round(speed_kmh)),
            "steering_deg": round(steering_deg, 1),
            "action": action,
            "alert_level": alert_level,
            "min_obs_dist": round(min_obs_dist, 1),
            "features": features_count,
        }
        self.log_entries.append(entry)

    def save_csv(self, filename: str = "trajectory_log.csv") -> str:
        """Export all recorded frames to CSV."""
        filepath = os.path.join(self.output_dir, filename)
        if not self.log_entries:
            return filepath

        fieldnames = list(self.log_entries[0].keys())
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.log_entries)

        return filepath

    def generate_plot(self, filename: str = "trajectory_comparison.png") -> Tuple[str, str]:
        """
        Generate trajectory and steering analysis plot.
        Returns: (saved_filepath, base64_image_data).
        """
        filepath = os.path.join(self.output_dir, filename)

        if not self.log_entries:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No video trajectory data recorded yet", ha="center", va="center")
            plt.tight_layout()
            plt.savefig(filepath, dpi=120)
            plt.close(fig)
            return filepath, ""

        times = [e["time_sec"] for e in self.log_entries]
        vo_x = [e["vo_x"] for e in self.log_entries]
        vo_y = [e["vo_y"] for e in self.log_entries]
        speeds = [e["speed_kmh"] for e in self.log_entries]
        steers = [e["steering_deg"] for e in self.log_entries]
        obs_dists = [min(50.0, e["min_obs_dist"]) for e in self.log_entries]

        total_distance = np.hypot(vo_x[-1] - vo_x[0], vo_y[-1] - vo_y[0])
        max_speed = max(speeds) if speeds else 0.0
        avg_speed = np.mean(speeds) if speeds else 0.0
        total_time = times[-1] if times else 0.0

        fig = plt.figure(figsize=(13, 7), dpi=130)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0], height_ratios=[1.0, 1.0])

        # Subplot 1: Reconstructed 2D Vehicle Trajectory (Visual Odometry)
        ax_map = fig.add_subplot(gs[:, 0])
        ax_map.set_title("Visual Odometry: Reconstructed Car Trajectory (GPS-Denied)", fontsize=12, fontweight="bold", pad=10)
        ax_map.set_xlabel("Lateral Position X (meters)", fontsize=10)
        ax_map.set_ylabel("Forward Position Y (meters)", fontsize=10)
        ax_map.grid(True, linestyle="--", alpha=0.5)

        # Plot path
        ax_map.plot(vo_x, vo_y, color="#2563eb", linewidth=2.5, linestyle="-", label="Reconstructed VO Path")
        # Start marker
        ax_map.scatter([vo_x[0]], [vo_y[0]], color="#16a34a", s=140, zorder=5, marker="o", label="Start Location")
        # Current / Final marker
        ax_map.scatter([vo_x[-1]], [vo_y[-1]], color="#dc2626", s=160, zorder=5, marker="*", label="End Location")

        # Mark obstacle warning events along path
        for e in self.log_entries:
            if e["alert_level"] == "CRITICAL":
                ax_map.scatter([e["vo_x"]], [e["vo_y"]], color="#ef4444", s=30, zorder=4, marker="x")

        ax_map.legend(loc="upper left", fontsize=9)
        ax_map.axis("equal")

        # Subplot 2: Vehicle Speed Profile
        ax_spd = fig.add_subplot(gs[0, 1])
        ax_spd.set_title("Estimated Vehicle Speed (km/h)", fontsize=10, fontweight="bold")
        ax_spd.plot(times, speeds, color="#059669", linewidth=2.0)
        ax_spd.set_xlabel("Time (s)", fontsize=9)
        ax_spd.set_ylabel("Speed (km/h)", fontsize=9)
        ax_spd.grid(True, linestyle="--", alpha=0.5)

        # Subplot 3: Steering Recommendation Profile
        ax_str = fig.add_subplot(gs[1, 1])
        ax_str.set_title("Recommended Steering Angle (deg)", fontsize=10, fontweight="bold")
        ax_str.plot(times, steers, color="#d97706", linewidth=1.8, label="Steering Angle")
        ax_str.axhline(0, color="#64748b", linestyle="--", alpha=0.7)
        ax_str.set_xlabel("Time (s)", fontsize=9)
        ax_str.set_ylabel("Steer Angle (deg)", fontsize=9)
        ax_str.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(filepath, dpi=130)

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        buf.seek(0)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        return filepath, b64_str
