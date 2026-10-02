"""
Random convex polygon generation for 2D rooms.
"""

import numpy as np


# Placement distance constraints (meters)
SOURCE_WALL_MIN_DIST = 0.5
MIC_WALL_MIN_DIST = 0.3
MIC_SOURCE_MIN_DIST = 0.5


def generate_polygon(n_seg, side_min, side_max, rng):
    """
    Generate a random convex polygon with n_seg vertices.

    Args:
        n_seg: Number of vertices/segments
        side_min, side_max: Range for room extent (meters)
        rng: numpy random generator

    Returns:
        vertices: (n_seg, 2) array
    """
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_seg))
    verts = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    extent = verts.max(0) - verts.min(0)
    target = rng.uniform(side_min, side_max)
    verts = verts * (target / max(extent))
    verts -= verts.min(0)
    verts += 0.3  # margin so room isn't on grid edge
    return verts


def polygon_area(vertices):
    """Compute area of a polygon via the shoelace formula."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_in_convex(vertices, pt):
    """Check if a point is inside a convex polygon."""
    n = len(vertices)
    sign = None
    for i in range(n):
        j = (i + 1) % n
        cross = np.cross(vertices[j] - vertices[i], pt - vertices[i])
        if sign is None:
            sign = cross >= 0
        elif (cross >= 0) != sign:
            return False
    return True


def point_to_segment_distance(pt, v1, v2):
    """Minimum distance from point to line segment."""
    pt, v1, v2 = np.asarray(pt), np.asarray(v1), np.asarray(v2)
    seg = v2 - v1
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-10:
        return np.linalg.norm(pt - v1)
    t = max(0, min(1, np.dot(pt - v1, seg) / seg_len_sq))
    proj = v1 + t * seg
    return np.linalg.norm(pt - proj)


def min_distance_to_walls(pt, vertices):
    """Minimum distance from point to any wall segment."""
    n = len(vertices)
    return min(
        point_to_segment_distance(pt, vertices[i], vertices[(i + 1) % n])
        for i in range(n)
    )


def sample_inside_with_margin(vertices, rng, min_wall_dist, max_tries=1000):
    """
    Sample a point inside a convex polygon with minimum wall distance.

    Returns centroid as fallback if no valid point found.
    """
    lo, hi = vertices.min(0), vertices.max(0)
    for _ in range(max_tries):
        pt = rng.uniform(lo, hi)
        if point_in_convex(vertices, pt):
            if min_distance_to_walls(pt, vertices) >= min_wall_dist:
                return pt
    return vertices.mean(0)


def sample_mic_with_constraints(vertices, rng, source_pos, min_wall_dist,
                                min_source_dist, max_tries=1000):
    """
    Sample a mic position with wall and source distance constraints.

    Returns best-effort position if no perfect solution found.
    """
    lo, hi = vertices.min(0), vertices.max(0)
    source_pos = np.asarray(source_pos)
    best_pt = None
    best_score = -float("inf")

    for _ in range(max_tries):
        pt = rng.uniform(lo, hi)
        if not point_in_convex(vertices, pt):
            continue
        wall_dist = min_distance_to_walls(pt, vertices)
        source_dist = np.linalg.norm(pt - source_pos)
        if wall_dist >= min_wall_dist and source_dist >= min_source_dist:
            return pt
        score = min(wall_dist / min_wall_dist, source_dist / min_source_dist)
        if score > best_score:
            best_score = score
            best_pt = pt

    return best_pt if best_pt is not None else vertices.mean(0)


def generate_world(rng, n_seg_min=3, n_seg_max=10, side_min=2.0,
                   side_max=8.0, kappa_min=0.1, kappa_max=10.0, c=343.0,
                   source_wall_margin=SOURCE_WALL_MIN_DIST):
    """
    Generate a random 2D convex polygon room with Robin BC.

    Returns dict with vertices, segments, kappas, source_pos, room_area, c.
    """
    n_seg = rng.integers(n_seg_min, n_seg_max + 1)
    vertices = generate_polygon(n_seg, side_min, side_max, rng)
    kappas = rng.uniform(kappa_min, kappa_max, size=n_seg)
    source_pos = sample_inside_with_margin(vertices, rng, source_wall_margin)
    segments = [[i, (i + 1) % n_seg] for i in range(n_seg)]
    area = polygon_area(vertices)

    return {
        "vertices": vertices.tolist(),
        "n_segments": int(n_seg),
        "segments": segments,
        "kappa_per_segment": kappas.tolist(),
        "source_pos": source_pos.tolist(),
        "room_area": float(area),
        "c": float(c),
    }


def place_mics(vertices, n_mics, rng, source_pos=None,
               mic_wall_margin=MIC_WALL_MIN_DIST,
               mic_source_margin=MIC_SOURCE_MIN_DIST):
    """
    Place microphones inside the room with distance constraints.

    Args:
        vertices: Room polygon vertices (N, 2)
        n_mics: Number of microphones
        rng: Random generator
        source_pos: Source position (for min distance constraint)
        mic_wall_margin: Min distance from walls (default 0.3m)
        mic_source_margin: Min distance from source (default 0.5m)

    Returns:
        (n_mics, 2) array of mic positions in meters
    """
    verts = np.asarray(vertices)
    if source_pos is None:
        return np.array([
            sample_inside_with_margin(verts, rng, mic_wall_margin)
            for _ in range(n_mics)
        ])

    source_pos = np.asarray(source_pos)
    return np.array([
        sample_mic_with_constraints(
            verts, rng, source_pos,
            min_wall_dist=mic_wall_margin,
            min_source_dist=mic_source_margin
        )
        for _ in range(n_mics)
    ])
