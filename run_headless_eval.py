"""
Headless Simulation & Evaluation Runner
Simulates the UGV camera feed, vehicle kinematics, and obstacles purely in Python.
Runs multiple trials, logs trajectory data, and generates the Matplotlib comparison plot.
Useful for automated testing, CI, or batch performance analysis.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from modules.perception import PerceptionDetector
from modules.visual_odometry import VisualOdometry
from modules.planner import PathPlanner
from modules.logger import TrajectoryLogger


class HeadlessSimulator:
    def __init__(self, course_length: float = 22.0, course_width: float = 14.0):
        self.start_pose = (0.0, 0.0)
        self.goal_pose = (course_length, course_width)
        self.goal_tolerance = 1.2

        # Static obstacles (x, y, radius)
        self.obstacles = [
          {"x": 5.5, "y": 2.5, "radius": 1.1},
          {"x": 10.0, "y": 6.8, "radius": 1.2},
          {"x": 13.5, "y": 5.2, "radius": 1.0},
          {"x": 16.0, "y": 11.2, "radius": 1.2},
          {"x": 18.5, "y": 9.5, "radius": 1.1},
        ]

    def render_virtual_camera_frame(
        self,
        gt_x: float,
        gt_y: float,
        gt_theta: float,
        width: int = 320,
        height: int = 240,
    ) -> np.ndarray:
        """
        Synthesize a realistic camera frame from current vehicle pose and obstacles.
        Simulates ground plane texture, horizon, and perspective-projected obstacle projections.
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Sky
        horizon_y = int(height * 0.35)
        frame[:horizon_y, :] = [235, 206, 135]  # Sky blue

        # 2. Outdoor ground (textured earthy olive-brown)
        frame[horizon_y:, :] = [92, 112, 107]

        # Add ground texture noise based on vehicle x, y coordinates
        np.random.seed(int((gt_x * 100 + gt_y * 50) % 10000))
        noise = np.random.randint(-12, 12, (height - horizon_y, width, 3))
        frame[horizon_y:, :] = np.clip(frame[horizon_y:, :] + noise, 0, 255).astype(np.uint8)

        # 3. Project static obstacles into vehicle camera perspective
        fov = np.radians(72)
        cam_height = 0.8  # meters above ground

        for obs in self.obstacles:
            dx = obs["x"] - gt_x
            dy = obs["y"] - gt_y

            # Transform to vehicle body frame
            forward_dist = dx * np.cos(gt_theta) + dy * np.sin(gt_theta)
            lateral_dist = -dx * np.sin(gt_theta) + dy * np.cos(gt_theta)

            # Check if obstacle is in front of camera (0.8m to 25m)
            if 0.8 < forward_dist < 22.0:
                # Perspective projection
                angle_h = np.arctan2(lateral_dist, forward_dist)

                if abs(angle_h) < (fov / 2):
                    # Horizontal pixel coordinate
                    u = int(width / 2 + (lateral_dist / (forward_dist * np.tan(fov / 2))) * (width / 2))

                    # Vertical pixel coordinate (closer -> lower in frame)
                    v_ground = int(horizon_y + (cam_height / forward_dist) * (height * 0.95))
                    v_ground = min(height - 2, max(horizon_y, v_ground))

                    # Projected obstacle dimensions
                    proj_r = int((obs["radius"] / forward_dist) * (width * 0.7))
                    proj_w = max(10, proj_r * 2)
                    proj_h = max(8, int(proj_w * 0.8))

                    top_y = max(horizon_y, v_ground - proj_h)
                    left_x = max(0, u - proj_w // 2)
                    right_x = min(width - 1, u + proj_w // 2)
                    bottom_y = min(height - 1, v_ground)

                    if right_x > left_x and bottom_y > top_y:
                        # Draw obstacle body (dark slate/rock color)
                        frame[top_y:bottom_y, left_x:right_x] = [45, 52, 58]
                        # Outline
                        cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (25, 30, 35), 2)

        return frame


def run_evaluation(num_runs: int = 1, max_steps: int = 400, output_dir: str = "logs"):
    os.makedirs(output_dir, exist_ok=True)
    sim = HeadlessSimulator()

    print("=" * 70)
    print(f" STARTING HEADLESS EVALUATION ({num_runs} RUN(S))")
    print("=" * 70)

    summary_results = []

    for run_idx in range(1, num_runs + 1):
        print(f"\n--- [RUN {run_idx}/{num_runs}] ---")

        # Instantiate modules
        perception = PerceptionDetector()
        vo = VisualOdometry()
        planner = PathPlanner(
            goal_x=sim.goal_pose[0],
            goal_y=sim.goal_pose[1],
            goal_tolerance=sim.goal_tolerance,
        )
        logger = TrajectoryLogger(output_dir=output_dir)

        # Initialize poses
        gt_x, gt_y, gt_theta = 0.0, 0.0, 0.0
        vo.reset(x=0.0, y=0.0, theta=0.0)
        logger.start_run(
            start_x=0.0,
            start_y=0.0,
            goal_x=sim.goal_pose[0],
            goal_y=sim.goal_pose[1],
            goal_tolerance=sim.goal_tolerance,
            obstacles=sim.obstacles,
        )

        dt = 0.05
        last_v = 0.0
        success = False

        for step in range(1, max_steps + 1):
            time_sec = step * dt

            # 1. Render simulated camera frame
            frame = sim.render_virtual_camera_frame(gt_x, gt_y, gt_theta)

            # 2. Perception: Ground vs Obstacle detection
            p_res = perception.process_frame(frame)
            densities = p_res["sector_densities"]
            obstacles = p_res["obstacles"]

            # 3. Visual Odometry (GPS-denied)
            vo_res = vo.process_frame(frame, nominal_velocity=last_v, dt=dt)
            vo_pose = vo_res["pose"]

            # 4. Reactive Path Planner
            plan = planner.plan_step(vo_pose, densities, obstacles)
            v = plan["linear_velocity"]
            w = plan["angular_velocity"]
            status = plan["status"]
            dist_to_goal = plan["distance_to_goal"]
            last_v = v

            # 5. Log step
            gt_pose = {"x": gt_x, "y": gt_y, "theta": gt_theta}
            logger.log_step(step, time_sec, gt_pose, vo_pose, v, w, status, dist_to_goal)

            # Print pipeline progress every 20 steps
            if step % 20 == 0 or status == "GOAL_REACHED":
                print(
                    f"Step {step:03d} | Time: {time_sec:5.1f}s | "
                    f"GT: ({gt_x:5.2f}, {gt_y:5.2f}) | "
                    f"VO: ({vo_pose['x']:5.2f}, {vo_pose['y']:5.2f}) | "
                    f"GoalDist: {dist_to_goal:4.1f}m | "
                    f"Cmd: v={v:.2f}, w={w:+.2f} | {status}"
                )

            # 6. Update Ground Truth Kinematics
            gt_theta = (gt_theta + w * dt + np.pi) % (2 * np.pi) - np.pi
            gt_x += v * np.cos(gt_theta) * dt
            gt_y += v * np.sin(gt_theta) * dt

            # Check goal condition
            actual_dist_to_goal = np.hypot(sim.goal_pose[0] - gt_x, sim.goal_pose[1] - gt_y)
            if actual_dist_to_goal <= sim.goal_tolerance or status == "GOAL_REACHED":
                success = True
                print(f"\n>>> MISSION SUCCESS: Goal reached at step {step} ({time_sec:.1f}s)! Actual dist: {actual_dist_to_goal:.2f}m")
                break

        # Save run outputs
        csv_path = logger.save_csv(f"trajectory_log_run_{run_idx}.csv")
        plot_path, _ = logger.generate_plot(f"trajectory_comparison_run_{run_idx}.png")

        summary_results.append({
            "run": run_idx,
            "success": success,
            "steps": step,
            "time_sec": step * dt,
            "final_gt_dist": actual_dist_to_goal,
            "csv": csv_path,
            "plot": plot_path,
        })

    print("\n" + "=" * 70)
    print(" EVALUATION SUMMARY ACROSS ALL RUNS")
    print("=" * 70)
    for res in summary_results:
        outcome = "SUCCESS" if res["success"] else "INCOMPLETE"
        print(f"Run {res['run']}: {outcome} | Duration: {res['time_sec']:.1f}s | Final Distance: {res['final_gt_dist']:.2f}m | Plot: {res['plot']}")

    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run headless UGV navigation evaluation")
    parser.add_argument("--runs", type=int, default=1, help="Number of simulation runs")
    parser.add_argument("--steps", type=int, default=350, help="Max steps per run")
    args = parser.parse_args()

    run_evaluation(num_runs=args.runs, max_steps=args.steps)
