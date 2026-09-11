import math


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def bbox_width(bbox):
    x1, _, x2, _ = bbox
    return x2 - x1


def bbox_height(bbox):
    _, y1, _, y2 = bbox
    return y2 - y1


def bbox_area(bbox):
    width = max(0.0, bbox_width(bbox))
    height = max(0.0, bbox_height(bbox))

    return width * height


def intersection_bbox(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def intersection_area(box_a, box_b):

    intersection = intersection_bbox(box_a, box_b)

    if intersection is None:
        return 0.0

    return bbox_area(intersection)


def iou(box_a, box_b):

    intersection = intersection_area(box_a, box_b)

    union = (
        bbox_area(box_a)
        + bbox_area(box_b)
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def center_distance(box_a, box_b):

    center_a = bbox_center(box_a)
    center_b = bbox_center(box_b)

    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    return math.sqrt(dx * dx + dy * dy)


def horizontal_distance(box_a, box_b):

    center_a = bbox_center(box_a)
    center_b = bbox_center(box_b)

    return abs(center_b[0] - center_a[0])


def vertical_distance(box_a, box_b):

    center_a = bbox_center(box_a)
    center_b = bbox_center(box_b)

    return abs(center_b[1] - center_a[1])


def relative_position(box_a, box_b):

    """
    Returns the position of B relative to A.

    Coordinate system:
        x increases → right
        y increases → down
    """

    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)

    dx = bx - ax
    dy = by - ay

    horizontal = "same"
    vertical = "same"

    if dx > 0:
        horizontal = "right"
    elif dx < 0:
        horizontal = "left"

    if dy > 0:
        vertical = "below"
    elif dy < 0:
        vertical = "above"

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "dx": dx,
        "dy": dy,
    }


def angle_degrees(box_a, box_b):

    """
    Angle from object A to object B.

    0°   = right
    90°  = down
    -90° = up
    180° = left
    """

    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)

    dx = bx - ax
    dy = by - ay

    return math.degrees(math.atan2(dy, dx))


def overlaps(box_a, box_b, threshold=0.0):

    return iou(box_a, box_b) > threshold


def relative_geometry(box_a, box_b):

    position = relative_position(box_a, box_b)

    return {
        "center_a": bbox_center(box_a),
        "center_b": bbox_center(box_b),
        "distance_pixels": center_distance(box_a, box_b),
        "horizontal_distance_pixels":
            horizontal_distance(box_a, box_b),
        "vertical_distance_pixels":
            vertical_distance(box_a, box_b),
        "angle_degrees":
            angle_degrees(box_a, box_b),
        "iou":
            iou(box_a, box_b),
        "horizontal_relation":
            position["horizontal"],
        "vertical_relation":
            position["vertical"],
    }
