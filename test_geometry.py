from geometry.spatial import (
    bbox_center,
    bbox_area,
    iou,
    center_distance,
    relative_position,
    angle_degrees,
    relative_geometry,
)


person = (100, 100, 200, 300)
bus = (400, 150, 800, 500)
near_person = (210, 100, 310, 300)


print("=" * 60)
print("GEOMETRY ENGINE TEST")
print("=" * 60)


print("\nPERSON")
print("BBox:", person)
print("Center:", bbox_center(person))
print("Area:", bbox_area(person))


print("\nBUS")
print("BBox:", bus)
print("Center:", bbox_center(bus))
print("Area:", bbox_area(bus))


print("\nPERSON -> BUS")

print(
    "Relative position:",
    relative_position(person, bus)
)

print(
    "Center distance:",
    center_distance(person, bus)
)

print(
    "Angle:",
    angle_degrees(person, bus)
)

print(
    "IoU:",
    iou(person, bus)
)


print("\nPERSON -> NEAR PERSON")

print(
    "Relative position:",
    relative_position(person, near_person)
)

print(
    "Distance:",
    center_distance(person, near_person)
)

print(
    "Angle:",
    angle_degrees(person, near_person)
)

print(
    "IoU:",
    iou(person, near_person)
)


print("\nCOMPLETE GEOMETRY")

print(
    relative_geometry(person, bus)
)


print("\nTEST COMPLETE")
