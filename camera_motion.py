import math
import cv2
import numpy as np


class CameraMotionEstimator:
    def __init__(
        self,
        max_features=1500,
        min_matches=10,
        ratio_threshold=0.75,
        ransac_threshold=3.0,
    ):
        self.max_features = max_features
        self.min_matches = min_matches
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold

        self.orb = cv2.ORB_create(
            nfeatures=max_features
        )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

        self.previous_gray = None
        self.cumulative_rotation_deg = 0.0

    def reset(self):
        self.previous_gray = None
        self.cumulative_rotation_deg = 0.0

    def estimate(self, frame):
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        keypoints, descriptors = self.orb.detectAndCompute(
            gray,
            None,
        )

        if (
            self.previous_gray is None
            or descriptors is None
            or len(keypoints) < self.min_matches
        ):
            self.previous_gray = gray

            return {
                "valid": False,
                "rotation_deg": 0.0,
                "translation_x": 0.0,
                "translation_y": 0.0,
                "scale": 1.0,
                "matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "cumulative_rotation_deg":
                    self.cumulative_rotation_deg,
                "affine_matrix": None,
            }

        previous_keypoints, previous_descriptors = (
            self.orb.detectAndCompute(
                self.previous_gray,
                None,
            )
        )

        if (
            previous_descriptors is None
            or len(previous_keypoints) < self.min_matches
        ):
            self.previous_gray = gray

            return {
                "valid": False,
                "rotation_deg": 0.0,
                "translation_x": 0.0,
                "translation_y": 0.0,
                "scale": 1.0,
                "matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "cumulative_rotation_deg":
                    self.cumulative_rotation_deg,
                "affine_matrix": None,
            }

        raw_matches = self.matcher.knnMatch(
            previous_descriptors,
            descriptors,
            k=2,
        )

        good_matches = []

        for pair in raw_matches:
            if len(pair) != 2:
                continue

            m, n = pair

            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.min_matches:
            self.previous_gray = gray

            return {
                "valid": False,
                "rotation_deg": 0.0,
                "translation_x": 0.0,
                "translation_y": 0.0,
                "scale": 1.0,
                "matches": len(good_matches),
                "inliers": 0,
                "inlier_ratio": 0.0,
                "cumulative_rotation_deg":
                    self.cumulative_rotation_deg,
                "affine_matrix": None,
            }

        src_points = np.float32([
            previous_keypoints[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_points = np.float32([
            keypoints[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        affine_matrix, inlier_mask = cv2.estimateAffinePartial2D(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=2000,
            confidence=0.99,
        )

        self.previous_gray = gray

        if affine_matrix is None or inlier_mask is None:
            return {
                "valid": False,
                "rotation_deg": 0.0,
                "translation_x": 0.0,
                "translation_y": 0.0,
                "scale": 1.0,
                "matches": len(good_matches),
                "inliers": 0,
                "inlier_ratio": 0.0,
                "cumulative_rotation_deg":
                    self.cumulative_rotation_deg,
                "affine_matrix": None,
            }

        inliers = int(inlier_mask.sum())
        matches = len(good_matches)

        a = float(affine_matrix[0, 0])
        b = float(affine_matrix[1, 0])

        rotation_rad = math.atan2(b, a)
        rotation_deg = math.degrees(rotation_rad)

        scale = math.sqrt(a * a + b * b)

        tx = float(affine_matrix[0, 2])
        ty = float(affine_matrix[1, 2])

        self.cumulative_rotation_deg += rotation_deg

        return {
            "valid": True,
            "rotation_deg": rotation_deg,
            "translation_x": tx,
            "translation_y": ty,
            "scale": scale,
            "matches": matches,
            "inliers": inliers,
            "inlier_ratio": inliers / matches
                if matches > 0 else 0.0,
            "cumulative_rotation_deg":
                self.cumulative_rotation_deg,
            "affine_matrix":
                affine_matrix.astype(np.float32),
        }
