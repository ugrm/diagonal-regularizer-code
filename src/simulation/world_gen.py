"""
Random 2D convex polygon rooms (G1) with Robin BC (M1).

Distance constraints:
- Source must be >= 0.5m from all walls
- Mics must be >= 0.3m from all walls
- Mics must be >= 0.5m from source
"""
import numpy as np


# Distance constraints (meters)
SOURCE_WALL_MIN_DIST = 0.5
MIC_WALL_MIN_DIST = 0.3
MIC_SOURCE_MIN_DIST = 0.5


def _random_convex_polygon(n_seg, side_min, side_max, rng):
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_seg))
    verts = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    extent = verts.max(0) - verts.min(0)
    target = rng.uniform(side_min, side_max)
    verts = verts * (target / max(extent))
    verts -= verts.min(0)
    verts += 0.3  # margin so room isn't on grid edge
    return verts


def _point_in_convex(vertices, pt):
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


def _point_to_segment_distance(pt, v1, v2):
    """Compute minimum distance from point to line segment."""
    pt = np.asarray(pt)
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Vector from v1 to v2
    seg = v2 - v1
    seg_len_sq = np.dot(seg, seg)

    if seg_len_sq < 1e-10:
        # Degenerate segment (v1 == v2)
        return np.linalg.norm(pt - v1)

    # Project pt onto line, clamped to segment
    t = max(0, min(1, np.dot(pt - v1, seg) / seg_len_sq))
    proj = v1 + t * seg

    return np.linalg.norm(pt - proj)


def _min_distance_to_walls(pt, vertices):
    """Compute minimum distance from point to any wall segment."""
    n = len(vertices)
    min_dist = float('inf')

    for i in range(n):
        j = (i + 1) % n
        dist = _point_to_segment_distance(pt, vertices[i], vertices[j])
        min_dist = min(min_dist, dist)

    return min_dist


def _sample_inside(vertices, rng, max_tries=500):
    """Sample a point inside the polygon (no distance constraint)."""
    lo, hi = vertices.min(0), vertices.max(0)
    for _ in range(max_tries):
        pt = rng.uniform(lo, hi)
        if _point_in_convex(vertices, pt):
            return pt
    return vertices.mean(0)


def _sample_inside_with_wall_margin(vertices, rng, min_wall_dist, max_tries=1000):
    """
    Sample a point inside the polygon with minimum distance from walls.

    Args:
        vertices: Polygon vertices
        rng: Random generator
        min_wall_dist: Minimum distance from any wall (meters)
        max_tries: Maximum sampling attempts

    Returns:
        Point satisfying constraint, or centroid if not found
    """
    lo, hi = vertices.min(0), vertices.max(0)

    for _ in range(max_tries):
        pt = rng.uniform(lo, hi)
        if _point_in_convex(vertices, pt):
            dist = _min_distance_to_walls(pt, vertices)
            if dist >= min_wall_dist:
                return pt

    # Fallback: return centroid (usually satisfies constraint for convex polygons)
    return vertices.mean(0)


def _sample_mic_with_constraints(vertices, rng, source_pos, min_wall_dist, min_source_dist, max_tries=1000):
    """
    Sample a mic position with wall and source distance constraints.

    Args:
        vertices: Polygon vertices
        rng: Random generator
        source_pos: Source position to maintain distance from
        min_wall_dist: Minimum distance from any wall (meters)
        min_source_dist: Minimum distance from source (meters)
        max_tries: Maximum sampling attempts

    Returns:
        Point satisfying all constraints, or best-effort fallback
    """
    lo, hi = vertices.min(0), vertices.max(0)
    source_pos = np.asarray(source_pos)

    best_pt = None
    best_score = -float('inf')

    for _ in range(max_tries):
        pt = rng.uniform(lo, hi)
        if not _point_in_convex(vertices, pt):
            continue

        wall_dist = _min_distance_to_walls(pt, vertices)
        source_dist = np.linalg.norm(pt - source_pos)

        # Check both constraints
        if wall_dist >= min_wall_dist and source_dist >= min_source_dist:
            return pt

        # Track best attempt for fallback
        score = min(wall_dist / min_wall_dist, source_dist / min_source_dist)
        if score > best_score:
            best_score = score
            best_pt = pt

    # Return best attempt if no perfect solution found
    if best_pt is not None:
        return best_pt

    # Last resort: offset from centroid
    return vertices.mean(0)


def generate_world(rng, n_seg_min=3, n_seg_max=10,
                   side_min=2.0, side_max=8.0,
                   kappa_min=0.1, kappa_max=10.0, c=343.0,
                   source_wall_margin=SOURCE_WALL_MIN_DIST):
    """
    Generate a random 2D convex polygon room with Robin BC.

    Args:
        rng: numpy random generator
        n_seg_min, n_seg_max: Range for number of wall segments
        side_min, side_max: Range for room side length (meters)
        kappa_min, kappa_max: Range for Robin BC coefficient per wall
            - dataset spec: [0.1, 10.0]
            - Lower κ = more reflective (harder wall)
            - Higher κ = more absorptive (softer wall)
        c: Speed of sound (m/s)
        source_wall_margin: Minimum source distance from walls (default 0.5m per spec)

    Returns:
        dict with:
            vertices: List of [x, y] corner positions
            n_segments: Number of wall segments
            segments: List of [v0, v1] vertex index pairs
            kappas: Robin BC coefficient per segment (backward compat)
            kappa_per_segment: Same as kappas
            source_pos: [x, y] source position (>= 0.5m from walls)
            room_area: Room area in m²
            c: Speed of sound
    """
    n_seg = rng.integers(n_seg_min, n_seg_max + 1)
    vertices = _random_convex_polygon(n_seg, side_min, side_max, rng)
    kappas = rng.uniform(kappa_min, kappa_max, size=n_seg)

    # Source must be >= 0.5m from walls
    source_pos = _sample_inside_with_wall_margin(vertices, rng, source_wall_margin)

    # Compute segments array [[v0, v1], [v1, v2], ..., [vn-1, v0]]
    segments = [[i, (i + 1) % n_seg] for i in range(n_seg)]

    # Compute room area using Shoelace formula
    x = vertices[:, 0]
    y = vertices[:, 1]
    room_area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return dict(
        vertices=vertices.tolist(),
        n_segments=int(n_seg),
        segments=segments,
        kappas=kappas.tolist(),  # Backward compatibility
        kappa_per_segment=kappas.tolist(),  # alias of kappas
        source_pos=source_pos.tolist(),
        source_position=source_pos.tolist(),  # alias of source_pos
        room_area=float(room_area),
        c=float(c),
    )


def place_mics(vertices, n_mics, rng, source_pos=None,
               mic_wall_margin=MIC_WALL_MIN_DIST,
               mic_source_margin=MIC_SOURCE_MIN_DIST):
    """
    Place microphones inside the room with distance constraints.

    Constraints:
    - Mics >= 0.3m from all walls
    - Mics >= 0.5m from source
    - Mics spread across room (not clustered)

    Args:
        vertices: Room polygon vertices
        n_mics: Number of microphones (typically 8)
        rng: Random generator
        source_pos: Source position (for min distance constraint)
        mic_wall_margin: Min distance from walls (default 0.3m)
        mic_source_margin: Min distance from source (default 0.5m)

    Returns:
        (n_mics, 2) array of mic positions in meters
    """
    verts = np.asarray(vertices)

    if source_pos is None:
        # No source constraint, just wall margin
        return np.array([
            _sample_inside_with_wall_margin(verts, rng, mic_wall_margin)
            for _ in range(n_mics)
        ])

    source_pos = np.asarray(source_pos)

    # Sample mics with both wall and source constraints
    mics = []
    for _ in range(n_mics):
        mic = _sample_mic_with_constraints(
            verts, rng, source_pos,
            min_wall_dist=mic_wall_margin,
            min_source_dist=mic_source_margin
        )
        mics.append(mic)

    return np.array(mics)