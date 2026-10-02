"""Room validation and degenerate room detection."""

import numpy as np

from .polygon import polygon_area


def detect_degenerate_rooms(rooms_manifest, min_area_fraction=0.05):
    """
    Detect rooms that are degenerate (too little interior area).

    Args:
        rooms_manifest: List of room metadata dicts
        min_area_fraction: Minimum interior-to-bounding-box ratio

    Returns:
        List of scene IDs that are degenerate
    """
    degenerate = []
    for room in rooms_manifest:
        area = room.get("room_area", 0.0)
        verts = np.array(room.get("vertices", []))
        if len(verts) < 3:
            degenerate.append(room["scene_id"])
            continue
        bbox_area = np.prod(verts.max(0) - verts.min(0))
        if bbox_area > 0 and area / bbox_area < min_area_fraction:
            degenerate.append(room["scene_id"])
    return degenerate
