"""
High-Accuracy Visual Odometry Module
Uses 5-Point Essential Matrix Decomposition with RANSAC & Lucas-Kanade Optical Flow
for robust monocular ego-motion estimation in GPS-denied road and off-road driving.
"""

from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


class VisualOdometry:
    def __init__(
        self,
        max_features: int = 160,
        feature_quality: float = 0.02,
        min_feature_distance: int = 12,
        camera_height: float = 1.20,
        speed_scale_factor: float = 0.18,
        speed_sensitivity: float = 1.65,
        yaw_scale_factor: float = 0.0018,
        focal_length: float = 440.0,
    ):
        self.max_features = max_features
        self.feature_quality = feature_quality
        self.min_feature_distance = min_feature_distance
        self.camera_height = camera_height
        self.speed_scale = speed_scale_factor
        self.speed_sensitivity = speed_sensitivity
        self.yaw_scale = yaw_scale_factor
        self.focal = focal_length

        # Pinhole camera intrinsic matrix (standard front dashcam model @ 640x360)
        self.K = np.array([
            [self.focal, 0.0, 320.0],
            [0.0, self.focal, 180.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=feature_quality,
            minDistance=min_feature_distance,
            blockSize=7,
        )

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.015),
        )

        # Pose state
        self.x: float = 0.0
        self.y: float = 0.0
        self.theta: float = 0.0
        self.total_distance: float = 0.0
        self.estimated_speed_kmh: float = 0.0

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.total_distance = 0.0
        self.estimated_speed_kmh = 0.0
        self.prev_gray = None
        self.prev_pts = None

    def _extract_features(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        mask = np.zeros_like(gray)
        # Focus feature extraction on rigid road surface and lower road borders (30% to 90%)
        mask[int(h * 0.30):int(h * 0.90), :] = 255
        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        return pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)

    def _estimate_step_distance(self, prev_pts: np.ndarray, next_pts: np.ndarray, h: int) -> float:
        """
        Estimate physically realistic forward translation (in meters) from optical flow
        using pinhole camera ground-plane perspective projection with horizon dampening:
            dZ = (f_y * H_cam) / (y - c_y)^2 * dy * sensitivity
        """
        flow_vectors = next_pts - prev_pts
        dy_pixels = flow_vectors[:, 1]

        # Ground features are situated in the drivable surface (y > h * 0.50)
        ground_mask = prev_pts[:, 1] > (h * 0.50)

        if np.sum(ground_mask) >= 3:
            g_y = prev_pts[ground_mask, 1]
            g_dy = dy_pixels[ground_mask]

            f_H = self.focal * self.camera_height
            c_y = self.K[1, 2]

            # Positive downward flow corresponds to forward vehicle translation
            forward_mask = g_dy > 0.12
            if np.sum(forward_mask) >= 2:
                valid_y = g_y[forward_mask]
                valid_dy = g_dy[forward_mask]
                # Enforce baseline distance from optical center (>= 32px) while preserving responsiveness
                effective_y_dist = np.maximum(32.0, valid_y - c_y)
                delta_z_arr = (f_H / (effective_y_dist ** 2)) * valid_dy * self.speed_sensitivity
                # Filter out extreme noise spikes (> 1.20 m/frame corresponds to > 130 km/h at 30 fps)
                inlier_dz = delta_z_arr[delta_z_arr < 1.20]
                if len(inlier_dz) > 0:
                    med_dz = float(np.median(inlier_dz))
                    return min(0.65, max(0.0, med_dz))
            elif np.median(g_dy) <= 0.12:
                # Vehicle is stationary or idling
                return 0.0

        # Robust fallback using median optical flow and calibrated scale factor
        pos_flow = max(0.0, float(np.median(dy_pixels)))
        if pos_flow < 0.12:
            return 0.0
        fallback_dz = pos_flow * self.speed_scale * self.speed_sensitivity
        return min(0.65, max(0.0, fallback_dz))

    def process_frame(
        self,
        frame: np.ndarray,
        dt: float = 0.033,
    ) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        flow_frame = frame.copy()

        delta_x = 0.0
        delta_y = 0.0
        delta_theta = 0.0
        step_dist = 0.0

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 15:
            self.prev_gray = gray
            self.prev_pts = self._extract_features(gray)
            return {
                "pose": {
                    "x": round(self.x, 2),
                    "y": round(self.y, 2),
                    "theta": round(self.theta, 3),
                    "theta_deg": round(np.degrees(self.theta), 2),
                },
                "speed_kmh": int(round(self.estimated_speed_kmh)),
                "total_distance": int(round(self.total_distance)),
                "tracked_features_count": len(self.prev_pts),
                "flow_frame": flow_frame,
            }

        # Pyramidal Lucas-Kanade Optical Flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        if next_pts is not None and status is not None:
            valid = (status.flatten() == 1)
            good_prev = self.prev_pts[valid].reshape(-1, 2)
            good_next = next_pts[valid].reshape(-1, 2)
        else:
            good_prev = np.empty((0, 2))
            good_next = np.empty((0, 2))

        tracked_count = len(good_next)
        used_essential_matrix = False

        if tracked_count >= 12:
            try:
                # Epipolar Geometry: Essential Matrix Estimation with RANSAC
                # Rejects non-rigid outliers and dynamic moving objects automatically
                E, inlier_mask = cv2.findEssentialMat(
                    good_next,
                    good_prev,
                    self.K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.2,
                )

                if E is not None and inlier_mask is not None and np.sum(inlier_mask) >= 8:
                    inliers = inlier_mask.ravel() == 1
                    clean_prev = good_prev[inliers]
                    clean_next = good_next[inliers]

                    # Recover Relative Camera Rotation (R) and Translation direction (t)
                    _, R, t, _ = cv2.recoverPose(E, clean_next, clean_prev, self.K)

                    # Extract Yaw rotation angle from R matrix (rotation about camera vertical Y axis)
                    # For forward camera: yaw = atan2(R[0, 2], R[2, 2])
                    yaw_rot = float(np.arctan2(R[0, 2], R[2, 2]))
                    # Limit extreme angle jumps
                    delta_theta = float(np.clip(yaw_rot, -0.08, 0.08))

                    # Scale translation magnitude using ground-plane perspective optical flow
                    step_dist = self._estimate_step_distance(clean_prev, clean_next, h)
                    used_essential_matrix = True

                    self.prev_pts = clean_next.reshape(-1, 1, 2)

                    # Draw inlier tracks on flow frame
                    for p0, p1 in zip(clean_prev, clean_next):
                        x0, y0 = int(p0[0]), int(p0[1])
                        x1, y1 = int(p1[0]), int(p1[1])
                        cv2.circle(flow_frame, (x1, y1), 3, (0, 255, 0), -1)
                        cv2.line(flow_frame, (x0, y0), (x1, y1), (0, 255, 255), 1)

            except Exception as e:
                used_essential_matrix = False

        # Fallback: Median/IQR Statistical Optical Flow Filter
        if not used_essential_matrix and tracked_count >= 6:
            flow_vectors = good_next - good_prev
            dx_pixels = flow_vectors[:, 0]
            dy_pixels = flow_vectors[:, 1]

            med_dx = float(np.median(dx_pixels))
            med_dy = float(np.median(dy_pixels))

            inliers = (
                (np.abs(dx_pixels - med_dx) < 2.5 * max(1.2, np.std(dx_pixels))) &
                (np.abs(dy_pixels - med_dy) < 2.5 * max(1.2, np.std(dy_pixels)))
            )

            if np.sum(inliers) >= 5:
                clean_dx = dx_pixels[inliers]
                clean_dy = dy_pixels[inliers]
                clean_prev = good_prev[inliers]
                clean_next = good_next[inliers]

                delta_theta = -float(np.median(clean_dx)) * self.yaw_scale

                step_dist = self._estimate_step_distance(clean_prev, clean_next, h)
                self.prev_pts = clean_next.reshape(-1, 1, 2)

                for p0, p1 in zip(clean_prev, clean_next):
                    x0, y0 = int(p0[0]), int(p0[1])
                    x1, y1 = int(p1[0]), int(p1[1])
                    cv2.circle(flow_frame, (x1, y1), 3, (0, 255, 0), -1)
                    cv2.line(flow_frame, (x0, y0), (x1, y1), (0, 255, 255), 1)
            else:
                self.prev_pts = good_next.reshape(-1, 1, 2)

        # Update speed (km/h) with responsive filtering
        speed_mps = step_dist / max(0.001, dt)
        current_speed_kmh = speed_mps * 3.6
        if step_dist <= 0.001:
            self.estimated_speed_kmh = max(0.0, self.estimated_speed_kmh * 0.60)
            if self.estimated_speed_kmh < 0.4:
                self.estimated_speed_kmh = 0.0
        else:
            self.estimated_speed_kmh = 0.72 * self.estimated_speed_kmh + 0.28 * current_speed_kmh

        # Planar ego-motion integration (forward along Y, lateral along X)
        mid_theta = self.theta + (delta_theta / 2.0)
        delta_x = step_dist * np.sin(mid_theta)
        delta_y = step_dist * np.cos(mid_theta)

        self.theta = (self.theta + delta_theta + np.pi) % (2 * np.pi) - np.pi
        self.x += delta_x
        self.y += delta_y
        self.total_distance += step_dist

        # Replenish feature points when count drops
        if tracked_count < 40:
            new_pts = self._extract_features(gray)
            if len(new_pts) > 0:
                if self.prev_pts is not None and len(self.prev_pts) > 0:
                    self.prev_pts = np.vstack([self.prev_pts, new_pts[:(self.max_features - len(self.prev_pts))]])
                else:
                    self.prev_pts = new_pts

        self.prev_gray = gray

        # Telemetry HUD on flow image (integer speed without decimal point)
        hud = f"VO: ({self.x:+.1f}m, {self.y:+.1f}m) | Speed: {int(round(self.estimated_speed_kmh))} km/h | Hotspots: {tracked_count}"
        cv2.putText(flow_frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return {
            "pose": {
                "x": round(self.x, 2),
                "y": round(self.y, 2),
                "theta": round(self.theta, 3),
                "theta_deg": round(np.degrees(self.theta), 2),
            },
            "speed_kmh": int(round(self.estimated_speed_kmh)),
            "total_distance": int(round(self.total_distance)),
            "tracked_features_count": tracked_count,
            "flow_frame": flow_frame,
        }
