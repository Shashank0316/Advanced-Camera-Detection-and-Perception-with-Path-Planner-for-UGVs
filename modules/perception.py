"""
High-Accuracy Perception Module - Classical Computer Vision for Road & Off-Road Terrains
Features:
1. Multi-Patch Adaptive Ground Surface Modeling (Road & Off-Road Terrains).
2. Deep Surface Marking Suppression (White/Yellow Lines, Paint Dashes, Chalk Streaks).
3. 3D Volumetric Hazard Verification (Dimensions, Aspect Ratio, Solidity, Shadow Contrast).
4. Multi-Frame Temporal Obstacle Tracking (eliminates single-frame flickers and noise).
5. Strict upper bound physical constraints to reject entire-frame/road false positives.
"""

from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np


class TrackedObstacle:
    """Represents an obstacle tracked consistently across consecutive video frames."""

    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], distance_est: float, sector: str):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.smoothed_bbox = [float(v) for v in bbox]
        self.distance_est = distance_est
        self.smoothed_dist = distance_est
        self.sector = sector
        self.hits = 1
        self.misses = 0
        self.confirmed = False

    def update(self, new_bbox: Tuple[int, int, int, int], new_dist: float, sector: str):
        self.bbox = new_bbox
        # Exponential smoothing on bounding box (alpha = 0.40)
        for i in range(4):
            self.smoothed_bbox[i] = 0.60 * self.smoothed_bbox[i] + 0.40 * new_bbox[i]
        self.distance_est = new_dist
        self.smoothed_dist = 0.65 * self.smoothed_dist + 0.35 * new_dist
        self.sector = sector
        self.hits += 1
        self.misses = 0
        if self.hits >= 2:  # Confirmed after 2 consecutive matching frames
            self.confirmed = True

    def mark_missed(self):
        self.misses += 1


class ObstacleTracker:
    """Multi-frame temporal consistency tracker to eliminate false positives and flickers."""

    def __init__(self, max_misses: int = 2, max_distance_jump: float = 75.0):
        self.tracks: List[TrackedObstacle] = []
        self.next_id: int = 1
        self.max_misses = max_misses
        self.max_dist_jump = max_distance_jump

    def reset(self):
        self.tracks = []
        self.next_id = 1

    def update_tracks(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        matched_track_indices = set()
        matched_detection_indices = set()

        # Match current detections with existing tracks using centroid distance
        for d_idx, det in enumerate(detections):
            cx, cy = det["center"]
            best_track_idx = None
            min_dist = self.max_dist_jump

            for t_idx, track in enumerate(self.tracks):
                if t_idx in matched_track_indices:
                    continue
                tx = track.smoothed_bbox[0] + track.smoothed_bbox[2] / 2.0
                ty = track.smoothed_bbox[1] + track.smoothed_bbox[3] / 2.0
                dist = np.hypot(cx - tx, cy - ty)

                if dist < min_dist:
                    min_dist = dist
                    best_track_idx = t_idx

            if best_track_idx is not None:
                self.tracks[best_track_idx].update(
                    new_bbox=det["bbox"],
                    new_dist=det["distance_est"],
                    sector=det["sector"],
                )
                matched_track_indices.add(best_track_idx)
                matched_detection_indices.add(d_idx)

        # Increment misses for unmatched tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_indices:
                track.mark_missed()

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        # Create new candidate tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detection_indices:
                new_track = TrackedObstacle(
                    track_id=self.next_id,
                    bbox=det["bbox"],
                    distance_est=det["distance_est"],
                    sector=det["sector"],
                )
                self.next_id += 1
                self.tracks.append(new_track)

        # Return only confirmed tracks that have persisted across frames
        confirmed_obstacles: List[Dict[str, Any]] = []
        for track in self.tracks:
            if track.confirmed:
                sb = [int(round(v)) for v in track.smoothed_bbox]
                confirmed_obstacles.append({
                    "id": track.track_id,
                    "bbox": tuple(sb),
                    "center": (sb[0] + sb[2] // 2, sb[1] + sb[3] // 2),
                    "area": sb[2] * sb[3],
                    "distance_est": round(track.smoothed_dist, 1),
                    "sector": track.sector,
                    "label": "Obstacle",
                })

        return confirmed_obstacles


class PerceptionDetector:
    def __init__(
        self,
        min_obstacle_area: int = 380,
        horizon_ratio: float = 0.40,
        min_obstacle_width: int = 22,
        min_obstacle_height: int = 18,
    ):
        self.min_obstacle_area = min_obstacle_area
        self.horizon_ratio = horizon_ratio
        self.min_obs_w = min_obstacle_width
        self.min_obs_h = min_obstacle_height
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.tracker = ObstacleTracker()

    def reset(self):
        self.tracker.reset()

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        horizon_y = int(h * self.horizon_ratio)

        roi = frame[horizon_y:, :]
        roi_h, roi_w = roi.shape[:2]

        # 1. Color Spaces & Lighting Equalization
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv_roi)
        v_eq = self.clahe.apply(v_ch)
        hsv_eq = cv2.merge([h_ch, s_ch, v_eq])
        gray_roi = v_eq

        # 2. Complete Road Safety Line & Paint Masking
        # Detect white lines, weathered stripes, dashed paint, and yellow dividers
        white_marks_mask = cv2.inRange(
            hsv_eq,
            np.array([0, 0, 150], dtype=np.uint8),
            np.array([180, 78, 255], dtype=np.uint8),
        )
        yellow_marks_mask = cv2.inRange(
            hsv_eq,
            np.array([14, 40, 115], dtype=np.uint8),
            np.array([40, 255, 255], dtype=np.uint8),
        )
        road_markings_mask = cv2.bitwise_or(white_marks_mask, yellow_marks_mask)

        # Dilate line mask to fully envelope line boundary edges
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated_markings = cv2.dilate(road_markings_mask, dilate_kernel, iterations=1)

        # 3. Multi-Patch Ground Surface Sampling (On-Road & Off-Road)
        # Sample distinct zones in near-field drivable ground
        patches = [
            hsv_eq[int(roi_h * 0.75):int(roi_h * 0.95), int(roi_w * 0.40):int(roi_w * 0.60)],  # Center
            hsv_eq[int(roi_h * 0.75):int(roi_h * 0.95), int(roi_w * 0.22):int(roi_w * 0.38)],  # Center-Left
            hsv_eq[int(roi_h * 0.75):int(roi_h * 0.95), int(roi_w * 0.62):int(roi_w * 0.78)],  # Center-Right
            hsv_eq[int(roi_h * 0.85):int(roi_h * 0.98), int(roi_w * 0.30):int(roi_w * 0.70)],  # Near Base
        ]

        ground_masks = []
        for patch in patches:
            if patch.size > 0:
                p_mean = np.mean(patch, axis=(0, 1))
                p_std = np.std(patch, axis=(0, 1))

                h_tol = max(35, int(p_std[0] * 3.0))
                s_tol = max(55, int(p_std[1] * 3.0))
                v_tol = max(65, int(p_std[2] * 3.0))

                low_b = np.array([max(0, int(p_mean[0] - h_tol)), max(0, int(p_mean[1] - s_tol)), max(18, int(p_mean[2] - v_tol))], dtype=np.uint8)
                upp_b = np.array([min(180, int(p_mean[0] + h_tol)), min(255, int(p_mean[1] + s_tol)), min(255, int(p_mean[2] + v_tol))], dtype=np.uint8)
                ground_masks.append(cv2.inRange(hsv_eq, low_b, upp_b))

        # Union of sampled ground colors
        combined_ground = ground_masks[0]
        for gm in ground_masks[1:]:
            combined_ground = cv2.bitwise_or(combined_ground, gm)

        # Recognize broad off-road outdoor terrain spectrums:
        grass_mask = cv2.inRange(hsv_eq, np.array([35, 28, 35], dtype=np.uint8), np.array([88, 255, 235], dtype=np.uint8))
        dirt_mask = cv2.inRange(hsv_eq, np.array([6, 25, 55], dtype=np.uint8), np.array([30, 255, 215], dtype=np.uint8))

        # Navigable surface includes ground samples, off-road terrain, and painted road markings
        navigable_ground = cv2.bitwise_or(combined_ground, dilated_markings)
        navigable_ground = cv2.bitwise_or(navigable_ground, grass_mask)
        navigable_ground = cv2.bitwise_or(navigable_ground, dirt_mask)

        # 4. 3D Obstacle Edge & Shadow Contrast Detection
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 110)

        # Scrub edges belonging to painted road lines
        real_obstacle_edges = cv2.bitwise_and(edges, cv2.bitwise_not(dilated_markings))
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_edges = cv2.dilate(real_obstacle_edges, edge_kernel, iterations=1)

        # 5. Non-Ground Clustering & Corridor Masking
        non_ground = cv2.bitwise_not(navigable_ground)
        obstacle_raw = cv2.bitwise_or(non_ground, dilated_edges)

        # Exclude all painted road markings
        obstacle_raw = cv2.bitwise_and(obstacle_raw, cv2.bitwise_not(dilated_markings))

        # Driving corridor trapezoid
        corridor_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        trapezoid_pts = np.array([
            [int(roi_w * 0.12), roi_h],
            [int(roi_w * 0.88), roi_h],
            [int(roi_w * 0.65), int(roi_h * 0.10)],
            [int(roi_w * 0.35), int(roi_h * 0.10)],
        ], dtype=np.int32)
        cv2.fillPoly(corridor_mask, [trapezoid_pts], 255)

        # Morphological opening and closing: removes tiny line fragments, dashes, and ground noise
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        obstacle_roi = cv2.morphologyEx(obstacle_raw, cv2.MORPH_OPEN, clean_kernel)
        obstacle_roi = cv2.morphologyEx(obstacle_roi, cv2.MORPH_CLOSE, clean_kernel)
        obstacle_roi = cv2.bitwise_and(obstacle_roi, corridor_mask)

        # Traversable corridor
        traversable_roi = cv2.bitwise_and(cv2.bitwise_not(obstacle_roi), corridor_mask)

        # Full frame masks
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        obstacle_mask[horizon_y:, :] = obstacle_roi

        traversable_mask = np.zeros((h, w), dtype=np.uint8)
        traversable_mask[horizon_y:, :] = traversable_roi

        # 6. Obstacle Contours with Rigorous 3D Volumetric Verification
        contours, _ = cv2.findContours(obstacle_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_detections: List[Dict[str, Any]] = []

        max_obs_w = int(w * 0.50)
        max_obs_h = int(roi_h * 0.60)
        max_obs_area = int(w * roi_h * 0.20)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_obstacle_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Volumetric filter 0: Maximum physical dimensions (rejects whole-road false positives)
            if bw > max_obs_w or bh > max_obs_h or area > max_obs_area:
                continue

            # Volumetric filter 1: Minimum 3D dimensions
            if bw < self.min_obs_w or bh < self.min_obs_h:
                continue

            # Volumetric filter 2: Aspect Ratio (rejects thin stripes and flat pavement bars)
            aspect_ratio = bh / max(1.0, bw)
            inv_aspect = bw / max(1.0, bh)
            if aspect_ratio > 2.1 or inv_aspect > 3.2:
                continue

            # Volumetric filter 3: Paint/Road Line Color Density
            sub_markings = road_markings_mask[y:y + bh, x:x + bw]
            marking_ratio = np.count_nonzero(sub_markings) / max(1.0, bw * bh)
            if marking_ratio > 0.18:
                continue

            # Volumetric filter 4: Mean Brightness (White lines are bright, real obstacles have shading/shadows)
            sub_v = v_eq[y:y + bh, x:x + bw]
            sub_s = s_ch[y:y + bh, x:x + bw]
            if np.mean(sub_v) > 160 and np.mean(sub_s) < 75:
                continue

            # Volumetric filter 5: Solidity (Rocks, vehicles, crates are solid blocks)
            solidity = area / max(1.0, bw * bh)
            if solidity < 0.38:
                continue

            full_y = y + horizon_y
            cx = x + bw // 2
            cy = full_y + bh // 2

            norm_y = (full_y + bh - horizon_y) / max(1, (h - horizon_y))
            est_distance = max(1.5, 35.0 / (norm_y * 8.0 + 1.0))

            sector_idx = min(4, max(0, int((cx / w) * 5)))
            sectors = ["Far-Left", "Left", "Center", "Right", "Far-Right"]
            sector_name = sectors[sector_idx]

            candidate_detections.append({
                "bbox": (x, full_y, bw, bh),
                "center": (cx, cy),
                "area": int(area),
                "distance_est": round(float(est_distance), 1),
                "sector": sector_name,
                "label": "Obstacle",
            })

        # 7. Temporal Tracking (Requires 2+ consecutive frame persistence to confirm)
        confirmed_obstacles = self.tracker.update_tracks(candidate_detections)

        # 8. Annotated Visualization Frame
        annotated_frame = frame.copy()

        # Green corridor overlay
        green_overlay = np.zeros_like(frame)
        green_overlay[horizon_y:, :][traversable_roi > 0] = (0, 185, 0)
        cv2.addWeighted(green_overlay, 0.32, annotated_frame, 1.0, 0, annotated_frame)

        # Draw confirmed obstacles with track IDs and distance
        for obs in confirmed_obstacles:
            x, full_y, bw, bh = obs["bbox"]
            est_dist = obs["distance_est"]
            track_id = obs["id"]

            box_color = (0, 0, 255) if est_dist < 10.0 else ((0, 165, 255) if est_dist < 20.0 else (0, 255, 255))
            cv2.rectangle(annotated_frame, (x, full_y), (x + bw, full_y + bh), box_color, 2)
            cv2.putText(
                annotated_frame,
                f"#{track_id} Obstacle: {est_dist:.1f}m",
                (x, max(full_y - 6, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1,
                cv2.LINE_AA,
            )

        # 9. Sector Densities in Near Driving Zone
        near_roi_y = int(roi_h * 0.45)
        near_obstacles = obstacle_roi[near_roi_y:, :]
        near_h, near_w = near_obstacles.shape[:2]

        sector_densities = {}
        sector_names = ["far_left", "left", "center", "right", "far_right"]
        sec_w = near_w // 5

        for i, name in enumerate(sector_names):
            x1 = i * sec_w
            x2 = (i + 1) * sec_w if i < 4 else near_w
            sec_slice = near_obstacles[:, x1:x2]
            density = float(np.count_nonzero(sec_slice) / max(1, sec_slice.size))
            sector_densities[name] = round(density, 3)

        # Draw horizon guide
        cv2.line(annotated_frame, (0, horizon_y), (w, horizon_y), (255, 255, 0), 1)

        return {
            "traversable_mask": traversable_mask,
            "obstacle_mask": obstacle_mask,
            "obstacles": confirmed_obstacles,
            "sector_densities": sector_densities,
            "annotated_frame": annotated_frame,
        }
