from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import time


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str

    confidence: float

    bbox: tuple
    center: tuple

    first_seen: float
    last_seen: float

    visible: bool = True

    detection_history: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )

    trajectory: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )

    def update(
        self,
        bbox: tuple,
        center: tuple,
        confidence: float,
        timestamp: Optional[float] = None,
    ):
        if timestamp is None:
            timestamp = time.time()

        self.bbox = bbox
        self.center = center
        self.confidence = confidence
        self.last_seen = timestamp
        self.visible = True

        self.detection_history.append({
            "timestamp": timestamp,
            "bbox": bbox,
            "center": center,
            "confidence": confidence,
        })

        self.trajectory.append({
            "timestamp": timestamp,
            "center": center,
        })

    def mark_missing(self):
        self.visible = False


class ObjectMap:

    def __init__(self, max_missing_time=2.0):
        self.objects = {}
        self.max_missing_time = max_missing_time

    def update(self, detections, timestamp=None):

        if timestamp is None:
            timestamp = time.time()

        currently_visible = set()

        # --------------------------------------------------
        # Update objects detected in this frame
        # --------------------------------------------------

        for detection in detections:

            track_id = int(detection["track_id"])

            currently_visible.add(track_id)

            if track_id not in self.objects:

                obj = TrackedObject(
                    track_id=track_id,
                    class_id=int(detection["class_id"]),
                    class_name=detection["class_name"],
                    confidence=float(detection["confidence"]),
                    bbox=detection["bbox"],
                    center=detection["center"],
                    first_seen=timestamp,
                    last_seen=timestamp,
                )

                obj.detection_history.append({
                    "timestamp": timestamp,
                    "bbox": detection["bbox"],
                    "center": detection["center"],
                    "confidence": detection["confidence"],
                })

                obj.trajectory.append({
                    "timestamp": timestamp,
                    "center": detection["center"],
                })

                self.objects[track_id] = obj

            else:

                self.objects[track_id].update(
                    bbox=detection["bbox"],
                    center=detection["center"],
                    confidence=detection["confidence"],
                    timestamp=timestamp,
                )

        # --------------------------------------------------
        # Objects absent from current frame
        # --------------------------------------------------

        for track_id, obj in self.objects.items():

            if track_id not in currently_visible:

                # Not visible in THIS frame.
                obj.mark_missing()

        return self.get_visible_objects()

    def get_visible_objects(self):
        return [
            obj
            for obj in self.objects.values()
            if obj.visible
        ]

    def get_all_objects(self):
        return list(self.objects.values())

    def get_object(self, track_id):
        return self.objects.get(track_id)

    def unique_object_count(self):
        return len(self.objects)

    def visible_object_count(self):
        return len(self.get_visible_objects())

    def to_dict(self):

        result = {}

        for track_id, obj in self.objects.items():

            result[track_id] = {
                "track_id": obj.track_id,
                "class_id": obj.class_id,
                "class_name": obj.class_name,
                "confidence": obj.confidence,
                "bbox": obj.bbox,
                "center": obj.center,
                "first_seen": obj.first_seen,
                "last_seen": obj.last_seen,
                "visible": obj.visible,
                "trajectory": list(obj.trajectory),
            }

        return result