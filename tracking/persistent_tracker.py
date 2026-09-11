from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = bbox_area(a) + bbox_area(b) - inter

    if union <= 0:
        return 0.0

    return inter / union


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(np.hypot(ax - bx, ay - by))


def transform_bbox(bbox: BBox, affine_matrix: Optional[np.ndarray]) -> BBox:
    """
    Transform bbox using an affine matrix mapping PREVIOUS FRAME -> CURRENT FRAME.
    """
    if affine_matrix is None:
        return bbox

    matrix = np.asarray(affine_matrix, dtype=np.float32)

    if matrix.shape != (2, 3):
        return bbox

    x1, y1, x2, y2 = bbox

    points = np.array(
        [
            [[x1, y1]],
            [[x2, y1]],
            [[x2, y2]],
            [[x1, y2]],
        ],
        dtype=np.float32,
    )

    transformed = cv2.transform(points, matrix).reshape(-1, 2)

    nx1 = float(np.min(transformed[:, 0]))
    ny1 = float(np.min(transformed[:, 1]))
    nx2 = float(np.max(transformed[:, 0]))
    ny2 = float(np.max(transformed[:, 1]))

    return nx1, ny1, nx2, ny2


def extract_appearance(
    frame: Optional[np.ndarray],
    bbox: BBox,
    hist_size: Tuple[int, int] = (16, 8),
) -> Optional[np.ndarray]:
    """
    Lightweight appearance descriptor.

    HSV histogram:
      H = 16 bins
      S = 8 bins

    Normalized so it is relatively insensitive to object size.
    """
    if frame is None:
        return None

    if frame.ndim != 3 or frame.shape[2] != 3:
        return None

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Ignore a tiny border around the crop.
    ch, cw = crop.shape[:2]

    if ch >= 10 and cw >= 10:
        bx = max(1, int(cw * 0.05))
        by = max(1, int(ch * 0.05))
        crop = crop[by:ch - by, bx:cw - bx]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        list(hist_size),
        [0, 180, 0, 256],
    )

    hist = cv2.normalize(
        hist,
        hist,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_L2,
    )

    return hist.flatten().astype(np.float32)


def appearance_similarity(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
) -> Optional[float]:
    if a is None or b is None:
        return None

    if a.shape != b.shape:
        return None

    # Histogram correlation:
    # +1 = very similar
    #  0 = weak relationship
    # -1 = opposite
    score = cv2.compareHist(
        a.astype(np.float32),
        b.astype(np.float32),
        cv2.HISTCMP_CORREL,
    )

    # Convert approximately [-1, 1] -> [0, 1]
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


@dataclass
class PersistentTrack:
    persistent_id: int
    class_id: int
    class_name: str

    bbox: BBox
    center: Point
    confidence: float

    first_seen: float
    last_seen: float

    visible: bool = True

    state: str = "ACTIVE"

    confirmed: bool = False
    consecutive_hits: int = 1

    last_detector_id: Optional[int] = None

    frames_seen: int = 1
    missed_frames: int = 0

    detector_ids: set = field(default_factory=set)

    trajectory: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )

    confidence_history: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )

    appearance: Optional[np.ndarray] = None

    def update(
        self,
        bbox: BBox,
        confidence: float,
        timestamp: float,
        detector_id: Optional[int],
        appearance: Optional[np.ndarray],
        min_confirmed_hits: int,
    ):
        self.bbox = bbox
        self.center = bbox_center(bbox)
        self.confidence = float(confidence)

        self.last_seen = timestamp
        self.visible = True
        self.state = "ACTIVE"

        self.last_detector_id = detector_id

        self.frames_seen += 1
        self.missed_frames = 0
        self.consecutive_hits += 1

        if detector_id is not None:
            self.detector_ids.add(int(detector_id))

        self.trajectory.append(self.center)
        self.confidence_history.append(self.confidence)

        if self.consecutive_hits >= min_confirmed_hits:
            self.confirmed = True

        # Exponential appearance update.
        if appearance is not None:
            if self.appearance is None:
                self.appearance = appearance.copy()
            else:
                alpha = 0.20
                self.appearance = (
                    (1.0 - alpha) * self.appearance
                    + alpha * appearance
                )

                norm = np.linalg.norm(self.appearance)

                if norm > 1e-8:
                    self.appearance /= norm

    def mark_missing(self):
        self.visible = False
        self.missed_frames += 1
        self.consecutive_hits = 0


class PersistentTracker:
    """
    Persistent identity layer above YOLO/ByteTrack.

    YOLO remains authoritative for:
      - class
      - confidence
      - bbox
      - detector localization

    This class only maintains persistent identity.
    """

    def __init__(
        self,
        max_missing_frames: int = 30,
        max_center_distance: float = 180.0,
        min_iou: float = 0.05,
        min_confirmed_hits: int = 3,

        # Association weights.
        distance_weight: float = 0.40,
        iou_weight: float = 0.30,
        appearance_weight: float = 0.30,

        # Appearance is optional when no frame is supplied.
        appearance_gate: float = 0.15,

        # Lost tracks can be recovered during this window.
        max_reid_frames: Optional[int] = None,
    ):
        self.max_missing_frames = int(max_missing_frames)
        self.max_center_distance = float(max_center_distance)
        self.min_iou = float(min_iou)

        self.min_confirmed_hits = int(min_confirmed_hits)

        self.distance_weight = float(distance_weight)
        self.iou_weight = float(iou_weight)
        self.appearance_weight = float(appearance_weight)

        self.appearance_gate = float(appearance_gate)

        if max_reid_frames is None:
            max_reid_frames = max_missing_frames

        self.max_reid_frames = int(max_reid_frames)

        self.tracks: Dict[int, PersistentTrack] = {}

        self.next_persistent_id = 1

        self.frame_index = 0

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def update(
        self,
        detections: List[dict],
        timestamp: float = 0.0,
        affine_matrix: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
        camera_state: Optional[dict] = None,
    ) -> List[PersistentTrack]:
        """
        detections format:

        {
            "track_id": int,
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "bbox": (x1, y1, x2, y2)
        }

        Returns currently visible PersistentTrack objects.

        `camera_state` is accepted for compatibility with the existing
        pipeline/tests. If it contains an `affine_matrix`, that matrix
        is used for camera-motion compensation.
        """

        self.frame_index += 1

        # Backward-compatible camera state handling.
        # The camera motion estimator stores the affine transform under
        # `affine_matrix`.
        if affine_matrix is None and camera_state is not None:
            if isinstance(camera_state, dict):
                affine_matrix = camera_state.get("affine_matrix")

        prepared = []

        for det in detections:
            bbox = tuple(float(v) for v in det["bbox"])

            appearance = extract_appearance(
                frame,
                bbox,
            )

            prepared.append(
                {
                    "track_id": det.get("track_id"),
                    "class_id": int(det["class_id"]),
                    "class_name": str(det["class_name"]),
                    "confidence": float(det["confidence"]),
                    "bbox": bbox,
                    "appearance": appearance,
                }
            )

        # -----------------------------------------------------
        # Predict existing track positions using camera motion.
        # -----------------------------------------------------

        candidates = []

        for persistent_id, track in self.tracks.items():

            if track.state == "REMOVED":
                continue

            if track.missed_frames > self.max_reid_frames:
                continue

            predicted_bbox = transform_bbox(
                track.bbox,
                affine_matrix,
            )

            candidates.append(
                (
                    persistent_id,
                    track,
                    predicted_bbox,
                )
            )

        # -----------------------------------------------------
        # Build association candidates.
        # -----------------------------------------------------

        associations = []

        for det_index, det in enumerate(prepared):

            det_bbox = det["bbox"]

            for persistent_id, track, predicted_bbox in candidates:

                # Class is authoritative.
                if det["class_id"] != track.class_id:
                    continue

                distance = center_distance(
                    predicted_bbox,
                    det_bbox,
                )

                iou = bbox_iou(
                    predicted_bbox,
                    det_bbox,
                )

                # Adaptive distance gate.
                predicted_area = max(
                    bbox_area(predicted_bbox),
                    1.0,
                )

                object_scale = np.sqrt(predicted_area)

                distance_gate = max(
                    self.max_center_distance,
                    object_scale * 1.5,
                )

                if distance > distance_gate and iou < self.min_iou:
                    continue

                # Distance score.
                distance_score = max(
                    0.0,
                    1.0 - distance / max(distance_gate, 1.0),
                )

                # IoU score.
                iou_score = float(
                    np.clip(iou, 0.0, 1.0)
                )

                # Appearance score.
                appearance_score = appearance_similarity(
                    track.appearance,
                    det["appearance"],
                )

                if appearance_score is None:
                    # No appearance available.
                    total_score = (
                        0.60 * distance_score
                        + 0.40 * iou_score
                    )
                else:
                    # If appearance is strongly contradictory,
                    # reject the association unless geometry is
                    # extremely strong.
                    if (
                        appearance_score < self.appearance_gate
                        and iou < 0.30
                    ):
                        continue

                    total_score = (
                        self.distance_weight * distance_score
                        + self.iou_weight * iou_score
                        + self.appearance_weight * appearance_score
                    )

                associations.append(
                    (
                        total_score,
                        det_index,
                        persistent_id,
                        distance,
                        iou,
                        appearance_score,
                    )
                )

        # -----------------------------------------------------
        # Global one-to-one greedy assignment.
        #
        # Candidates are sorted by strongest association first.
        # Each detection and persistent track can be used once.
        # -----------------------------------------------------

        associations.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        matched_detections = set()
        matched_tracks = set()

        matches = []

        for association in associations:

            (
                score,
                det_index,
                persistent_id,
                distance,
                iou,
                appearance_score,
            ) = association

            if det_index in matched_detections:
                continue

            if persistent_id in matched_tracks:
                continue

            # Minimum association quality.
            if score < 0.20:
                continue

            matched_detections.add(det_index)
            matched_tracks.add(persistent_id)

            matches.append(
                (
                    det_index,
                    persistent_id,
                )
            )

        # -----------------------------------------------------
        # Update matched tracks.
        # -----------------------------------------------------

        for det_index, persistent_id in matches:

            det = prepared[det_index]
            track = self.tracks[persistent_id]

            track.update(
                bbox=det["bbox"],
                confidence=det["confidence"],
                timestamp=timestamp,
                detector_id=det["track_id"],
                appearance=det["appearance"],
                min_confirmed_hits=self.min_confirmed_hits,
            )

        # -----------------------------------------------------
        # Create new tracks for unmatched detections.
        # -----------------------------------------------------

        for det_index, det in enumerate(prepared):

            if det_index in matched_detections:
                continue

            persistent_id = self.next_persistent_id
            self.next_persistent_id += 1

            detector_id = det["track_id"]

            track = PersistentTrack(
                persistent_id=persistent_id,
                class_id=det["class_id"],
                class_name=det["class_name"],
                bbox=det["bbox"],
                center=bbox_center(det["bbox"]),
                confidence=det["confidence"],
                first_seen=timestamp,
                last_seen=timestamp,
                visible=True,
                state="ACTIVE",
                confirmed=(
                    self.min_confirmed_hits <= 1
                ),
                consecutive_hits=1,
                last_detector_id=detector_id,
                frames_seen=1,
                missed_frames=0,
                detector_ids=(
                    {int(detector_id)}
                    if detector_id is not None
                    else set()
                ),
                appearance=det["appearance"],
            )

            track.trajectory.append(track.center)
            track.confidence_history.append(track.confidence)

            self.tracks[persistent_id] = track

        # -----------------------------------------------------
        # Mark unmatched tracks as LOST.
        # -----------------------------------------------------

        for persistent_id, track in self.tracks.items():

            if track.state == "REMOVED":
                continue

            if persistent_id in matched_tracks:
                continue

            # Newly created tracks have already been seen this frame.
            if track.last_seen == timestamp and track.visible:
                continue

            track.mark_missing()

            if track.missed_frames > self.max_reid_frames:
                track.state = "REMOVED"
                track.visible = False

            else:
                track.state = "LOST"
                track.visible = False

        return self.get_visible_tracks()

    # ---------------------------------------------------------
    # Queries
    # ---------------------------------------------------------

    def get_visible_tracks(self) -> List[PersistentTrack]:
        return [
            track
            for track in self.tracks.values()
            if track.visible
            and track.state != "REMOVED"
        ]

    def get_active_tracks(self) -> List[PersistentTrack]:
        return [
            track
            for track in self.tracks.values()
            if track.state == "ACTIVE"
        ]

    def get_lost_tracks(self) -> List[PersistentTrack]:
        return [
            track
            for track in self.tracks.values()
            if track.state == "LOST"
        ]

    def get_track(
        self,
        persistent_id: int,
    ) -> Optional[PersistentTrack]:
        return self.tracks.get(persistent_id)

    def get_unique_count(self) -> int:
        """
        Number of persistent identities created during this
        tracker lifetime, including removed identities.
        """
        return len(self.tracks)

    def get_visible_count(self) -> int:
        return len(self.get_visible_tracks())

    def get_confirmed_visible_count(self) -> int:
        return sum(
            1
            for track in self.get_visible_tracks()
            if track.confirmed
        )

    def get_state_counts(self) -> Dict[str, int]:
        result = {
            "ACTIVE": 0,
            "LOST": 0,
            "REMOVED": 0,
        }

        for track in self.tracks.values():
            result[track.state] = (
                result.get(track.state, 0) + 1
            )

        return result

    def reset(self):
        self.tracks.clear()
        self.next_persistent_id = 1
        self.frame_index = 0
