import time

from tracking.object_map import ObjectMap


object_map = ObjectMap(max_missing_time=2.0)

t = time.time()

# Frame 1
detections = [
    {
        "track_id": 1,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": (100, 100, 200, 300),
        "center": (150, 200),
    },
    {
        "track_id": 2,
        "class_id": 5,
        "class_name": "bus",
        "confidence": 0.91,
        "bbox": (400, 150, 800, 500),
        "center": (600, 325),
    },
]

object_map.update(detections, timestamp=t)

print("FRAME 1")
print("Unique objects:", object_map.unique_object_count())
print("Visible objects:", object_map.visible_object_count())

# Frame 2
t += 0.1

detections = [
    {
        "track_id": 1,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.96,
        "bbox": (110, 105, 210, 305),
        "center": (160, 205),
    },
    {
        "track_id": 2,
        "class_id": 5,
        "class_name": "bus",
        "confidence": 0.92,
        "bbox": (405, 150, 805, 500),
        "center": (605, 325),
    },
]

object_map.update(detections, timestamp=t)

print("\nFRAME 2")
print("Unique objects:", object_map.unique_object_count())
print("Visible objects:", object_map.visible_object_count())

person = object_map.get_object(1)

print("\nPERSON ID 1")
print("Class:", person.class_name)
print("Center:", person.center)
print("Trajectory:", list(person.trajectory))

# Frame 3: only bus visible
t += 0.1

detections = [
    {
        "track_id": 2,
        "class_id": 5,
        "class_name": "bus",
        "confidence": 0.93,
        "bbox": (410, 150, 810, 500),
        "center": (610, 325),
    }
]

object_map.update(detections, timestamp=t)

print("\nFRAME 3")
print("Unique objects:", object_map.unique_object_count())
print("Visible objects:", object_map.visible_object_count())

print("\nALL OBJECTS")

for obj in object_map.get_all_objects():

    print(
        f"ID={obj.track_id} "
        f"class={obj.class_name} "
        f"visible={obj.visible} "
        f"trajectory_points={len(obj.trajectory)}"
    )

print("\nTEST COMPLETE")
