import numpy as np

from tracking.persistent_tracker import PersistentTracker


def make_detection(
    track_id,
    class_id,
    class_name,
    bbox,
    confidence=0.9,
):
    x1, y1, x2, y2 = bbox

    return {
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,
        "center": (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        ),
    }


def test_same_detector_id():
    tracker = PersistentTracker(
        max_missing_frames=5,
        max_center_distance=100,
        min_iou=0.05,
    )

    d1 = make_detection(
        1,
        0,
        "person",
        (100, 100, 200, 300),
    )

    tracks = tracker.update(
        [d1],
        timestamp=1.0,
    )

    assert len(tracks) == 1

    pid = tracks[0].persistent_id

    d2 = make_detection(
        1,
        0,
        "person",
        (105, 102, 205, 302),
    )

    tracks = tracker.update(
        [d2],
        timestamp=2.0,
    )

    assert len(tracks) == 1
    assert tracks[0].persistent_id == pid
    assert tracks[0].frames_seen == 2

    print("PASS: same detector ID")


def test_detector_id_changes():
    tracker = PersistentTracker(
        max_missing_frames=5,
        max_center_distance=100,
        min_iou=0.05,
    )

    d1 = make_detection(
        1,
        0,
        "person",
        (100, 100, 200, 300),
    )

    tracks = tracker.update(
        [d1],
        timestamp=1.0,
    )

    pid = tracks[0].persistent_id

    d2 = make_detection(
        77,
        0,
        "person",
        (104, 103, 204, 303),
    )

    tracks = tracker.update(
        [d2],
        timestamp=2.0,
    )

    assert len(tracks) == 1
    assert tracks[0].persistent_id == pid
    assert tracks[0].last_detector_id == 77
    assert 77 in tracks[0].detector_ids

    print("PASS: detector ID change")


def test_camera_translation():
    tracker = PersistentTracker(
        max_missing_frames=5,
        max_center_distance=150,
        min_iou=0.0,
    )

    d1 = make_detection(
        1,
        0,
        "person",
        (100, 100, 200, 300),
    )

    tracks = tracker.update(
        [d1],
        timestamp=1.0,
    )

    pid = tracks[0].persistent_id

    # Simulated camera translation:
    # object moves +50 pixels X and +20 pixels Y.
    affine = np.array(
        [
            [1.0, 0.0, 50.0],
            [0.0, 1.0, 20.0],
        ],
        dtype=np.float32,
    )

    camera_state = {
        "valid": True,
        "affine_matrix": affine,
    }

    d2 = make_detection(
        99,
        0,
        "person",
        (150, 120, 250, 320),
    )

    tracks = tracker.update(
        [d2],
        camera_state=camera_state,
        timestamp=2.0,
    )

    assert len(tracks) == 1
    assert tracks[0].persistent_id == pid
    assert tracks[0].last_detector_id == 99

    print("PASS: camera translation")


def test_camera_rotation():
    tracker = PersistentTracker(
        max_missing_frames=5,
        max_center_distance=150,
        min_iou=0.0,
    )

    d1 = make_detection(
        1,
        0,
        "person",
        (100, 100, 200, 200),
    )

    tracks = tracker.update(
        [d1],
        timestamp=1.0,
    )

    pid = tracks[0].persistent_id

    # 90-degree rotation around the origin.
    affine = np.array(
        [
            [0.0, -1.0, 400.0],
            [1.0,  0.0,   0.0],
        ],
        dtype=np.float32,
    )

    camera_state = {
        "valid": True,
        "affine_matrix": affine,
    }

    # Original corners:
    # (100,100), (200,100), (200,200), (100,200)
    #
    # After transform:
    # (300,100), (300,200), (200,200), (200,100)
    #
    # Resulting bbox:
    # (200,100,300,200)

    d2 = make_detection(
        50,
        0,
        "person",
        (200, 100, 300, 200),
    )

    tracks = tracker.update(
        [d2],
        camera_state=camera_state,
        timestamp=2.0,
    )

    assert len(tracks) == 1
    assert tracks[0].persistent_id == pid

    print("PASS: camera rotation")


def test_class_mismatch():
    tracker = PersistentTracker(
        max_missing_frames=5,
        max_center_distance=100,
        min_iou=0.05,
    )

    person = make_detection(
        1,
        0,
        "person",
        (100, 100, 200, 300),
    )

    tracks = tracker.update(
        [person],
        timestamp=1.0,
    )

    pid_person = tracks[0].persistent_id

    laptop = make_detection(
        2,
        63,
        "laptop",
        (100, 100, 200, 300),
    )

    tracks = tracker.update(
        [laptop],
        timestamp=2.0,
    )

    assert len(tracks) == 1
    assert tracks[0].persistent_id != pid_person

    print("PASS: class mismatch")


def test_multiple_objects():
    tracker = PersistentTracker(
        max_missing_frames=5,
        max_center_distance=120,
        min_iou=0.0,
    )

    detections = [
        make_detection(
            1,
            0,
            "person",
            (100, 100, 180, 280),
        ),
        make_detection(
            2,
            0,
            "person",
            (500, 100, 580, 280),
        ),
    ]

    tracks = tracker.update(
        detections,
        timestamp=1.0,
    )

    assert len(tracks) == 2

    pids = {
        track.persistent_id
        for track in tracks
    }

    assert len(pids) == 2

    # Detector IDs are changed/swapped.
    detections2 = [
        make_detection(
            22,
            0,
            "person",
            (505, 102, 585, 282),
        ),
        make_detection(
            11,
            0,
            "person",
            (102, 103, 182, 283),
        ),
    ]

    tracks = tracker.update(
        detections2,
        timestamp=2.0,
    )

    assert len(tracks) == 2

    assert {
        track.persistent_id
        for track in tracks
    } == pids

    print("PASS: multiple-object association")


def test_missing_and_recovery():
    tracker = PersistentTracker(
        max_missing_frames=3,
        max_center_distance=100,
        min_iou=0.0,
    )

    d1 = make_detection(
        1,
        0,
        "person",
        (100, 100, 200, 300),
    )

    tracks = tracker.update(
        [d1],
        timestamp=1.0,
    )

    pid = tracks[0].persistent_id

    # Object temporarily disappears.
    tracks = tracker.update(
        [],
        timestamp=2.0,
    )

    assert len(tracks) == 0

    # Same physical object returns with a new detector ID.
    d2 = make_detection(
        88,
        0,
        "person",
        (105, 105, 205, 305),
    )

    tracks = tracker.update(
        [d2],
        timestamp=3.0,
    )

    assert len(tracks) == 1
    assert tracks[0].persistent_id == pid

    print("PASS: temporary disappearance/recovery")


def main():
    test_same_detector_id()
    test_detector_id_changes()
    test_camera_translation()
    test_camera_rotation()
    test_class_mismatch()
    test_multiple_objects()
    test_missing_and_recovery()

    print()
    print("=" * 60)
    print("ALL PERSISTENT TRACKER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
