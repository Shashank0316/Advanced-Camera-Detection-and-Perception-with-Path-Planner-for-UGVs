"""
UGV Video Pipeline Controller (Optimized)
Coordinates:
Video Ingestion -> Perception -> Visual Odometry -> Steering Advisor -> Telemetry Logger.
Streamlined for low latency and minimal WebSocket overhead.
"""

import base64
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from .perception import PerceptionDetector
from .visual_odometry import VisualOdometry
from .planner import PathPlanner
from .logger import TrajectoryLogger
from .video_processor import VideoProcessor


class UGVController:
    def __init__(self, log_dir: str = "."):
        self.log_dir = log_dir
        self.perception = PerceptionDetector()
        self.vo = VisualOdometry()
        self.planner = PathPlanner()
        self.logger = TrajectoryLogger(output_dir=log_dir)
        self.video_processor = VideoProcessor(target_width=640, target_height=360)

        self.current_frame_idx: int = 0
        self.is_running: bool = False

    def reset(self):
        self.current_frame_idx = 0
        self.perception.reset()
        self.vo.reset()
        self.planner.reset()
        self.logger.start_run()
        self.is_running = True

    def load_video(self, source: Any) -> bool:
        ok = self.video_processor.open_source(source)
        if ok:
            self.reset()
        return ok

    def process_next_video_frame(self) -> Optional[Dict[str, Any]]:
        ok, frame, frame_idx = self.video_processor.read_frame()
        if not ok or frame is None:
            return None
        dt = 1.0 / max(10.0, min(120.0, self.video_processor.fps))
        return self.process_frame_image(frame, frame_idx=frame_idx, dt=dt)

    def process_frame_image(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
        dt: float = 0.033,
    ) -> Dict[str, Any]:
        if frame_idx is None:
            self.current_frame_idx += 1
            frame_idx = self.current_frame_idx
        else:
            self.current_frame_idx = frame_idx

        time_sec = frame_idx * dt

        # 1. Classical CV Perception
        p_res = self.perception.process_frame(frame)
        traversable_mask = p_res["traversable_mask"]
        obstacles = p_res["obstacles"]
        densities = p_res["sector_densities"]
        annotated_frame = p_res["annotated_frame"]

        # 2. Monocular Visual Odometry
        vo_res = self.vo.process_frame(frame, dt=dt)
        vo_pose = vo_res["pose"]
        flow_frame = vo_res["flow_frame"]
        speed_kmh = vo_res["speed_kmh"]
        total_dist = vo_res["total_distance"]
        features_count = vo_res["tracked_features_count"]

        # 3. Path Planner / Driver Assistance
        plan = self.planner.plan_step(
            current_pose=vo_pose,
            sector_densities=densities,
            detected_obstacles=obstacles,
        )
        steering_deg = plan["steering_angle_deg"]
        action = plan["recommended_action"]
        advisory = plan["advisory_message"]
        alert_level = plan["alert_level"]

        # 4. Log Step
        self.logger.log_step(
            frame_idx=frame_idx,
            time_sec=time_sec,
            vo_pose=vo_pose,
            speed_kmh=speed_kmh,
            steering_deg=steering_deg,
            action=action,
            alert_level=alert_level,
            min_obs_dist=plan["min_obstacle_dist"],
            features_count=features_count,
        )

        # 5. Optimized Base64 Image Encodings (Minimal payload size)
        # Main annotated frame (640x360 @ 65% quality)
        _, p_buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        p_b64 = base64.b64encode(p_buf).decode("utf-8")

        # Downsample secondary feeds to 320x180 for lightning fast transfer
        small_mask = cv2.resize(traversable_mask, (320, 180), interpolation=cv2.INTER_NEAREST)
        _, m_buf = cv2.imencode(".jpg", small_mask, [cv2.IMWRITE_JPEG_QUALITY, 50])
        m_b64 = base64.b64encode(m_buf).decode("utf-8")

        small_flow = cv2.resize(flow_frame, (320, 180))
        _, f_buf = cv2.imencode(".jpg", small_flow, [cv2.IMWRITE_JPEG_QUALITY, 60])
        f_b64 = base64.b64encode(f_buf).decode("utf-8")

        # Throttled console logging
        if frame_idx % 20 == 0 or alert_level == "CRITICAL":
            print(
                f"[F{frame_idx:04d}] "
                f"VO: ({vo_pose['x']:+5.1f}m, {vo_pose['y']:+5.1f}m) | "
                f"Spd: {int(round(speed_kmh)):3d}km/h | "
                f"Steer: {steering_deg:+4.1f}° | "
                f"Obs: {len(obstacles)} | {action}"
            )

        return {
            "frame": frame_idx,
            "time_sec": round(time_sec, 2),
            "vo_pose": vo_pose,
            "telemetry": {
                "speed_kmh": int(round(speed_kmh)),
                "total_distance_m": int(round(total_dist)),
                "steering_angle_deg": steering_deg,
                "action": action,
                "advisory": advisory,
                "alert_level": alert_level,
                "min_obstacle_dist": plan["min_obstacle_dist"],
                "obstacle_count": len(obstacles),
                "features_tracked": features_count,
            },
            "visualizations": {
                "perception_annotated": f"data:image/jpeg;base64,{p_b64}",
                "traversable_mask": f"data:image/jpeg;base64,{m_b64}",
                "optical_flow": f"data:image/jpeg;base64,{f_b64}",
            },
        }

    def finish_run(self) -> Dict[str, Any]:
        csv_path = self.logger.save_csv("trajectory_log.csv")
        plot_path, plot_b64 = self.logger.generate_plot("trajectory_comparison.png")
        self.is_running = False
        return {
            "csv_path": csv_path,
            "plot_path": plot_path,
            "plot_image": f"data:image/png;base64,{plot_b64}" if plot_b64 else None,
            "total_frames": self.current_frame_idx,
        }
