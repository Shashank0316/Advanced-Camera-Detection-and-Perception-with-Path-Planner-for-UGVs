"""
Video Ingestion & Stream Processing Module
Handles reading video files (MP4, AVI, MOV), live webcams, and generating
built-in demo dashcam footage for instant testing.
"""

import os
from typing import Optional, Tuple, Dict, Any, Generator
import cv2
import numpy as np


class VideoProcessor:
    def __init__(self, target_width: int = 640, target_height: int = 360):
        self.target_width = target_width
        self.target_height = target_height
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_path: str = ""
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.current_frame_idx: int = 0

    def open_source(self, source: Any) -> bool:
        """
        Open a video file path (str) or webcam device index (int).
        """
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open video source: {source}")
            return False

        self.source_path = str(source)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.current_frame_idx = 0
        print(f"[VIDEO] Successfully opened source '{source}' ({self.total_frames} frames @ {self.fps:.1f} FPS)")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        Read and resize the next frame.
        Returns: (success, frame_bgr, frame_index)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, 0

        ret, frame = self.cap.read()
        if not ret:
            return False, None, self.current_frame_idx

        self.current_frame_idx += 1

        # Resize to standard processing dimensions
        resized_frame = cv2.resize(frame, (self.target_width, self.target_height))
        return True, resized_frame, self.current_frame_idx

    def seek_frame(self, frame_number: int) -> bool:
        """Seek to a specific frame number."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame_idx = frame_number
            return True
        return False

    def restart(self):
        """Rewind video to beginning."""
        self.seek_frame(0)

    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source": self.source_path,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 1),
            "current_frame": self.current_frame_idx,
            "duration_sec": round(self.total_frames / max(1.0, self.fps), 1),
            "width": self.target_width,
            "height": self.target_height,
        }

    @staticmethod
    def generate_demo_dashcam_video(
        output_path: str,
        duration_sec: int = 12,
        fps: int = 30,
        width: int = 640,
        height: int = 360,
    ) -> str:
        """
        Generate a synthetic realistic outdoor road driving video clip
        with asphalt road, moving dashed lane markings, trees, roadside verges,
        and an approaching obstacle box for testing when no real video is supplied.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = duration_sec * fps
        horizon_y = int(height * 0.42)

        print(f"[DEMO] Synthesizing realistic outdoor driving clip ({duration_sec}s, {total_frames} frames)...")

        for frame_idx in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # 1. Sky Gradient (Light Blue to Pale Blue at horizon)
            for y in range(horizon_y):
                ratio = y / horizon_y
                b = int(235 * (1 - ratio * 0.2))
                g = int(200 * (1 - ratio * 0.15))
                r = int(140 + ratio * 40)
                frame[y, :] = [b, g, r]

            # 2. Roadside Grass/Verge (Lower half, dark earthy green)
            frame[horizon_y:, :] = [45, 85, 52]

            # Roadside distant tree line / hills
            cv2.ellipse(frame, (120, horizon_y), (140, 25), 0, 180, 360, (30, 60, 35), -1)
            cv2.ellipse(frame, (480, horizon_y), (160, 30), 0, 180, 360, (25, 55, 30), -1)

            # 3. Perspective Asphalt Road Surface (Dark slate gray trapezoid)
            road_top_w = 70
            road_bottom_w = 460
            road_pts = np.array([
                [width // 2 - road_top_w // 2, horizon_y],
                [width // 2 + road_top_w // 2, horizon_y],
                [width // 2 + road_bottom_w // 2, height],
                [width // 2 - road_bottom_w // 2, height],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [road_pts], (75, 78, 82))

            # Asphalt texture noise
            np.random.seed((frame_idx * 17) % 5000)
            road_noise = np.random.randint(-8, 8, (height - horizon_y, width, 3))
            road_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(road_mask, [road_pts], 255)
            frame[horizon_y:, :][road_mask[horizon_y:, :] > 0] = np.clip(
                frame[horizon_y:, :][road_mask[horizon_y:, :] > 0] + road_noise[road_mask[horizon_y:, :] > 0],
                0, 255
            ).astype(np.uint8)

            # 4. Moving Center Lane Dashes (simulates forward driving motion)
            dash_scroll = (frame_idx * 14) % 60
            for dy in range(0, height - horizon_y, 45):
                cur_y = horizon_y + dy + dash_scroll
                if cur_y >= height:
                    continue
                norm_pos = (cur_y - horizon_y) / (height - horizon_y)
                dash_len = int(8 + norm_pos * 30)
                dash_w = int(2 + norm_pos * 7)
                cx = width // 2
                cv2.line(frame, (cx, cur_y), (cx, min(height - 1, cur_y + dash_len)), (230, 230, 235), dash_w)

            # 5. Continuous White Road Edge Lines
            cv2.line(frame, (width // 2 - road_top_w // 2, horizon_y), (width // 2 - road_bottom_w // 2, height), (220, 220, 225), 3)
            cv2.line(frame, (width // 2 + road_top_w // 2, horizon_y), (width // 2 + road_bottom_w // 2, height), (220, 220, 225), 3)

            # 6. Moving Car Driving Ahead in Lane (Frames 50 to 330)
            if 50 <= frame_idx <= 330:
                t = (frame_idx - 50) / 280.0
                # Car approaches, cruises, and maintains distance
                car_progress = 0.18 + 0.52 * np.sin(t * np.pi * 0.9)
                car_y = int(horizon_y + 15 + car_progress * (height - horizon_y - 95))
                car_scale = 0.38 + car_progress * 0.85
                car_w = int(68 * car_scale)
                car_h = int(48 * car_scale)
                # Gentle lateral lane variation
                lateral_shift = int(np.sin(t * np.pi * 2.0) * 22 * car_scale)
                car_x = int(width // 2 - car_w // 2 + 35 + lateral_shift)

                # A. Undercarriage dark shadow (V < 52)
                shadow_pad = int(4 * car_scale)
                cv2.rectangle(frame, (car_x + shadow_pad, car_y + car_h - 4), (car_x + car_w - shadow_pad, car_y + car_h + 6), (15, 15, 18), -1)

                # B. Car Body (Metallic Slate Blue / Navy BGR: 155, 60, 45)
                cv2.rectangle(frame, (car_x, car_y + int(car_h * 0.28)), (car_x + car_w, car_y + car_h), (155, 60, 45), -1)
                cv2.rectangle(frame, (car_x, car_y + int(car_h * 0.28)), (car_x + car_w, car_y + car_h), (90, 30, 20), 1)

                # C. Cabin & Rear Windshield
                cabin_pts = np.array([
                    [car_x + int(car_w * 0.16), car_y + int(car_h * 0.28)],
                    [car_x + int(car_w * 0.26), car_y],
                    [car_x + int(car_w * 0.74), car_y],
                    [car_x + int(car_w * 0.84), car_y + int(car_h * 0.28)],
                ], dtype=np.int32)
                cv2.fillPoly(frame, [cabin_pts], (135, 50, 35))

                windshield_pts = np.array([
                    [car_x + int(car_w * 0.22), car_y + int(car_h * 0.26)],
                    [car_x + int(car_w * 0.30), car_y + int(car_h * 0.05)],
                    [car_x + int(car_w * 0.70), car_y + int(car_h * 0.05)],
                    [car_x + int(car_w * 0.78), car_y + int(car_h * 0.26)],
                ], dtype=np.int32)
                cv2.fillPoly(frame, [windshield_pts], (40, 40, 45))

                # D. Rear Bumper & License Plate
                lp_w = max(10, int(car_w * 0.28))
                lp_h = max(4, int(car_h * 0.14))
                lp_x = car_x + (car_w - lp_w) // 2
                lp_y = car_y + int(car_h * 0.65)
                cv2.rectangle(frame, (lp_x, lp_y), (lp_x + lp_w, lp_y + lp_h), (210, 210, 215), -1)

                # E. Bright Red Tail Lights
                tl_w = max(4, int(car_w * 0.16))
                tl_h = max(3, int(car_h * 0.18))
                tl_y = car_y + int(car_h * 0.36)
                cv2.rectangle(frame, (car_x + 3, tl_y), (car_x + 3 + tl_w, tl_y + tl_h), (0, 0, 245), -1)
                cv2.rectangle(frame, (car_x + car_w - 3 - tl_w, tl_y), (car_x + car_w - 3, tl_y + tl_h), (0, 0, 245), -1)

            writer.write(frame)

        writer.release()
        print(f"[DEMO] Road demo video successfully saved to: {output_path}")
        return output_path

    @staticmethod
    def generate_demo_offroad_video(
        output_path: str,
        duration_sec: int = 12,
        fps: int = 30,
        width: int = 640,
        height: int = 360,
    ) -> str:
        """
        Generate a synthetic off-road outdoor terrain driving video clip
        with dirt path, rugged grass banks, pebbles/stones, and an approaching 3D boulder hazard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = duration_sec * fps
        horizon_y = int(height * 0.40)

        print(f"[DEMO] Synthesizing outdoor off-road driving clip ({duration_sec}s, {total_frames} frames)...")

        for frame_idx in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # 1. Outdoor Sky Gradient (Sunlit afternoon)
            for y in range(horizon_y):
                ratio = y / horizon_y
                frame[y, :] = [int(225 - ratio * 30), int(195 - ratio * 20), int(135 + ratio * 40)]

            # 2. Rugged Off-Road Grass & Vegetation (Earthy dark green / olive)
            frame[horizon_y:, :] = [35, 75, 45]

            # Distant tree foliage & hills
            cv2.ellipse(frame, (160, horizon_y), (180, 35), 0, 180, 360, (25, 55, 30), -1)
            cv2.ellipse(frame, (460, horizon_y), (200, 40), 0, 180, 360, (20, 50, 25), -1)

            # 3. Off-Road Dirt Track (Earthy brown/ochre corridor)
            track_top_w = 80
            track_bottom_w = 480
            track_pts = np.array([
                [width // 2 - track_top_w // 2, horizon_y],
                [width // 2 + track_top_w // 2, horizon_y],
                [width // 2 + track_bottom_w // 2, height],
                [width // 2 - track_bottom_w // 2, height],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [track_pts], (52, 92, 122))  # Dirt brown (BGR)

            # Dirt surface texture noise
            np.random.seed((frame_idx * 13) % 4000)
            dirt_noise = np.random.randint(-10, 10, (height - horizon_y, width, 3))
            track_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(track_mask, [track_pts], 255)
            frame[horizon_y:, :][track_mask[horizon_y:, :] > 0] = np.clip(
                frame[horizon_y:, :][track_mask[horizon_y:, :] > 0] + dirt_noise[track_mask[horizon_y:, :] > 0],
                0, 255
            ).astype(np.uint8)

            # 4. Moving Small Flat Pebbles & Surface Chalk Streaks (Simulates forward vehicle motion)
            # These are flat on the ground and must NOT be flagged as obstacles!
            scroll_offset = (frame_idx * 12) % 70
            for dy in range(10, height - horizon_y - 20, 50):
                py = horizon_y + dy + scroll_offset
                if py >= height - 5:
                    continue
                norm = (py - horizon_y) / (height - horizon_y)
                pebble_r = int(2 + norm * 5)
                # Small white/gray pebble on left
                cv2.circle(frame, (int(width * 0.42 - norm * 30), py), pebble_r, (210, 215, 220), -1)
                # Flat chalk streak on right
                cv2.ellipse(frame, (int(width * 0.58 + norm * 35), py + 15), (int(pebble_r * 2.2), int(pebble_r * 0.8)), 15, 0, 360, (205, 210, 215), -1)

            # 5. Approaching 3D Boulder Hazard (Frames 80 to 260)
            if 80 <= frame_idx <= 260:
                progress = (frame_idx - 80) / 180.0
                obs_y = int(horizon_y + progress * (height - horizon_y - 50))
                obs_scale = 0.25 + progress * 1.05
                obs_w = int(55 * obs_scale)
                obs_h = int(45 * obs_scale)
                obs_x = int(width // 2 + (25 + progress * 65))

                # Real 3D Boulder with shadow
                # Ground contact shadow (dark)
                cv2.ellipse(frame, (obs_x + obs_w // 2, obs_y + obs_h), (obs_w // 2 + 5, 8), 0, 0, 360, (15, 18, 20), -1)
                # Boulder body (dark rugged slate)
                cv2.rectangle(frame, (obs_x, obs_y), (obs_x + obs_w, obs_y + obs_h), (38, 44, 48), -1)
                cv2.rectangle(frame, (obs_x, obs_y), (obs_x + obs_w, obs_y + obs_h), (20, 25, 28), 2)
                # Boulder facet highlight
                cv2.line(frame, (obs_x + 5, obs_y + 5), (obs_x + obs_w - 5, obs_y + 15), (60, 68, 75), 2)

            writer.write(frame)

        writer.release()
        print(f"[DEMO] Off-road demo video successfully saved to: {output_path}")
        return output_path
