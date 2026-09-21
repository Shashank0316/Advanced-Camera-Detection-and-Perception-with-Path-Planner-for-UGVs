"""
CLI Video Pipeline Runner
Process Dash Camera footage (or live webcam) directly from the terminal.
Generates an annotated video, trajectory CSV log, and Matplotlib trajectory plot.

Usage:
  python run_video_pipeline.py --video "path/to/car_footage.mp4" --show --save-video
  python run_video_pipeline.py --demo --show
  python run_video_pipeline.py --webcam 0 --show
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from modules.perception import PerceptionDetector
from modules.visual_odometry import VisualOdometry
from modules.planner import PathPlanner
from modules.logger import TrajectoryLogger
from modules.video_processor import VideoProcessor


def run_pipeline(
    video_path: Optional[str] = None,
    webcam_index: Optional[int] = None,
    use_demo: bool = False,
    show_window: bool = True,
    save_video: bool = True,
    output_dir: str = "logs",
):
    os.makedirs(output_dir, exist_ok=True)
    processor = VideoProcessor(target_width=640, target_height=360)

    # 1. Determine Video Source
    if use_demo or (video_path is None and webcam_index is None):
        demo_file = os.path.join(output_dir, "demo_dashcam.mp4")
        if not os.path.exists(demo_file):
            VideoProcessor.generate_demo_dashcam_video(demo_file, duration_sec=10, fps=30)
        source = demo_file
    elif webcam_index is not None:
        source = webcam_index
    else:
        source = video_path

    print("=" * 75)
    print(" VISION-BASED AUTONOMOUS NAVIGATION - CAR CAMERA PIPELINE")
    print(f" Source: {source}")
    print("=" * 75)

    if not processor.open_source(source):
        print(f"[ERROR] Could not open video source: {source}")
        return

    # 2. Instantiate modules
    perception = PerceptionDetector()
    vo = VisualOdometry()
    planner = PathPlanner()
    logger = TrajectoryLogger(output_dir=output_dir)
    logger.start_run()

    # 3. Setup Video Writer if requested
    writer = None
    output_video_path = os.path.join(output_dir, "annotated_output.mp4")
    if save_video:
        # Output side-by-side: Left = Perception Overlay, Right = Optical Flow
        out_w = 640 * 2
        out_h = 360
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, int(processor.fps), (out_w, out_h))
        print(f"[INFO] Saving processed video to: {output_video_path}")

    dt = 1.0 / max(1.0, processor.fps)
    frame_count = 0
    start_wall_time = time.time()

    try:
        while True:
            ret, frame, frame_idx = processor.read_frame()
            if not ret or frame is None:
                print("\n[INFO] End of video reached.")
                break

            frame_count += 1
            time_sec = frame_count * dt

            # Module 1: Classical CV Perception
            p_res = perception.process_frame(frame)
            obstacles = p_res["obstacles"]
            densities = p_res["sector_densities"]
            annotated_frame = p_res["annotated_frame"]
            traversable_mask = p_res["traversable_mask"]

            # Module 2: Monocular Visual Odometry (GPS-denied)
            vo_res = vo.process_frame(frame, dt=dt)
            vo_pose = vo_res["pose"]
            speed_kmh = vo_res["speed_kmh"]
            flow_frame = vo_res["flow_frame"]

            # Module 3: Path Planner & Steering Assistance
            plan = planner.plan_step(vo_pose, densities, obstacles)
            steering_deg = plan["steering_angle_deg"]
            action = plan["recommended_action"]
            alert_level = plan["alert_level"]

            # Module 4: Logging
            logger.log_step(
                frame_idx=frame_idx,
                time_sec=time_sec,
                vo_pose=vo_pose,
                speed_kmh=speed_kmh,
                steering_deg=steering_deg,
                action=action,
                alert_level=alert_level,
                min_obs_dist=plan["min_obstacle_dist"],
                features_count=vo_res["tracked_features_count"],
            )

            # Draw HUD banner on annotated frame
            hud_color = (0, 0, 255) if alert_level == "CRITICAL" else ((0, 165, 255) if alert_level == "CAUTION" else (0, 255, 0))
            cv2.putText(
                annotated_frame,
                f"ACTION: {action} (Steer: {steering_deg:+.1f} deg)",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                hud_color,
                2,
                cv2.LINE_AA,
            )

            # Combined Dashboard Display (Side-by-side)
            combined_display = np.hstack([annotated_frame, flow_frame])

            if writer is not None:
                writer.write(combined_display)

            if show_window:
                cv2.imshow("Car Camera Autonomous Navigation - Perception & Visual Odometry", combined_display)
                # Press 'q' or ESC to exit early
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:
                    print("\n[INFO] User interrupted playback.")
                    break

            if frame_count % 25 == 0:
                print(
                    f"Frame {frame_count:04d} | "
                    f"VO: ({vo_pose['x']:+5.1f}m, {vo_pose['y']:+5.1f}m) | "
                    f"Speed: {speed_kmh:4.1f} km/h | "
                    f"Steer: {steering_deg:+4.1f} deg | "
                    f"Obs: {len(obstacles)} | {action}"
                )

    finally:
        processor.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    # 4. Generate Trajectory Plot & CSV
    csv_file = logger.save_csv("trajectory_log.csv")
    plot_file, _ = logger.generate_plot("trajectory_comparison.png")

    total_time = time.time() - start_wall_time
    print("\n" + "=" * 75)
    print(" PROCESSING COMPLETE")
    print("=" * 75)
    print(f" Frames Processed: {frame_count} in {total_time:.1f}s ({frame_count / max(0.1, total_time):.1f} FPS)")
    print(f" Telemetry CSV:    {csv_file}")
    print(f" Trajectory Plot:  {plot_file}")
    if save_video:
        print(f" Output Video:     {output_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process real car camera footage for autonomous navigation")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file (MP4, AVI, MOV)")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam device index (e.g. 0)")
    parser.add_argument("--demo", action="store_true", help="Generate and run with demo outdoor driving video")
    parser.add_argument("--show", action="store_true", help="Show live visualization window")
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video")
    parser.add_argument("--outdir", type=str, default="logs", help="Output directory")

    args = parser.parse_args()
    run_pipeline(
        video_path=args.video,
        webcam_index=args.webcam,
        use_demo=args.demo,
        show_window=args.show,
        save_video=args.save_video,
        output_dir=args.outdir,
    )
