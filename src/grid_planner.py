"""Global path planning around walls on the 0.9 m TALL raster (the demo's wall-collision fix).

The demo used to route a walk as a STRAIGHT LINE from the current pose to the furniture, split into
<=1.1 m hops (planner/demo_end2end.expand_plan). If a wall sits between them, every hop aims at a
point across the wall and the body walks THROUGH it -- decode-time steering (guided_seg, RESULTS §13)
can only nudge within the model's own generative variance, it cannot invent a metre-scale detour.
This module computes a collision-free waypoint polyline first (A* on the inflated tall grid), so the
route goes AROUND walls and every resulting hop is short and wall-free; guided_seg then only cleans up
the model's local drift. CLAUDE.md §4: planning is "not excluded on principle".

Obstacles are the 0.9 m tall raster (walls survive, low furniture drops out -- the same map collision
is scored on, RESULTS §8), DILATED by a body radius so the ROOT path keeps clearance and the swinging
body does not graze. pixel<->world use the shared BEV extent (scene_anchors convention).
"""
import heapq

import numpy as np
from scipy import ndimage

from scene_anchors import world_to_px, px_to_world  # same extent convention as the demo

BODY_RADIUS_M = 0.28   # keep the root this far from a wall (torso half-width + margin)


def _disk(radius_cells):
    r = int(np.ceil(radius_cells))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= radius_cells * radius_cells


def free_space(tall, extent, inflate_m=BODY_RADIUS_M):
    """Boolean walkable map = NOT (tall dilated by inflate_m). Cells within inflate_m of a wall are
    blocked so the root path (and thus the body) keeps clearance."""
    H, W = tall.shape
    cell_m = 0.5 * ((extent[1] - extent[0]) / W + (extent[3] - extent[2]) / H)
    rad = max(1.0, inflate_m / cell_m)
    blocked = ndimage.binary_dilation(np.asarray(tall, bool), structure=_disk(rad))
    return ~blocked


# Clearance levels tried from most to least, so the path keeps as much wall clearance as the room
# allows. A cluttered room (scene0000) disconnects at 0.28 m but is one component at 0.20 m; a clean
# room stays connected at 0.28 m. Adaptive selection (build_levels + plan_path) avoids hand-tuning.
INFLATE_LEVELS = (0.28, 0.22, 0.17, 0.12)


def build_levels(tall, extent, inflates=INFLATE_LEVELS):
    """Precompute (inflate, free, labels) per clearance level once per scene, so plan_path can pick
    the largest clearance that connects a given start->goal without recomputing dilations per route."""
    out = []
    for infl in inflates:
        free = free_space(tall, extent, infl)
        lbl, _ = ndimage.label(free)
        out.append((infl, free, lbl))
    return out


def _nearest_free(free, rc, max_r=40):
    """Nearest free cell to rc (BFS rings) -- rescues a start/goal that landed in the inflated band."""
    if free[rc]:
        return rc
    H, W = free.shape
    r0, c0 = rc
    for rad in range(1, max_r):
        for dr in range(-rad, rad + 1):
            for dc in (-rad, rad) if abs(dr) != rad else range(-rad, rad + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < H and 0 <= c < W and free[r, c]:
                    return (r, c)
    return rc


_NEI = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.41421356), (-1, 1, 1.41421356), (1, -1, 1.41421356), (1, 1, 1.41421356)]


def _astar(free, start, goal):
    """8-connected A* on a boolean free grid. Returns a list of (r,c) cells or None. Diagonal moves
    are blocked from cutting a wall corner (both orthogonal neighbours must be free)."""
    H, W = free.shape
    if not free[start] or not free[goal]:
        return None

    def h(rc):
        dr, dc = abs(rc[0] - goal[0]), abs(rc[1] - goal[1])
        return 1.41421356 * min(dr, dc) + abs(dr - dc)

    openq = [(h(start), 0.0, start)]
    came, g = {start: None}, {start: 0.0}
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while came[cur] is not None:
                cur = came[cur]; path.append(cur)
            return path[::-1]
        if gc > g.get(cur, 1e18):
            continue
        r, c = cur
        for dr, dc, cost in _NEI:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or not free[nr, nc]:
                continue
            if dr != 0 and dc != 0 and not (free[r + dr, c] and free[r, c + dc]):
                continue  # don't slip diagonally between two wall cells
            ng = gc + cost
            if ng < g.get((nr, nc), 1e18):
                g[(nr, nc)] = ng; came[(nr, nc)] = cur
                heapq.heappush(openq, (ng + h((nr, nc)), ng, (nr, nc)))
    return None


def _los(free, a, b):
    """True if the straight cell-line a->b stays entirely in free space (Bresenham)."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err = dr - dc
    r, c = r0, c0
    while True:
        if not free[r, c]:
            return False
        if (r, c) == (r1, c1):
            return True
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc


def _simplify(path, free):
    """Line-of-sight shortcutting: keep a waypoint only where the straight shortcut breaks, so a
    staircase A* path becomes a few natural legs."""
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _los(free, path[i], path[j]):
            j -= 1
        out.append(path[j]); i = j
    return out


def _plan_on(free, start_xy, goal_xy, extent, shape):
    s = _nearest_free(free, world_to_px(np.asarray(start_xy, float), extent, shape))
    g = _nearest_free(free, world_to_px(np.asarray(goal_xy, float), extent, shape))
    cells = _astar(free, s, g)
    if cells is None:
        return None
    cells = _simplify(cells, free)
    wps = [px_to_world(rc, extent, shape) for rc in cells[1:]]
    if wps:
        wps[-1] = np.asarray(goal_xy, float)   # end exactly at the requested goal
    else:
        wps = [np.asarray(goal_xy, float)]
    return wps


def plan_path(start_xy, goal_xy, tall, extent, inflate_m=BODY_RADIUS_M, free=None, levels=None):
    """Collision-free world waypoints from start to goal AROUND walls, EXCLUDING the start and
    INCLUDING the goal. ADAPTIVE clearance: uses the largest INFLATE_LEVELS radius whose free space
    connects start and goal (a cluttered room disconnects at the widest clearance); falls back to
    the finest level, and finally to [goal] (straight line) so callers degrade gracefully.

    Pass precomputed `levels` (build_levels output) to reuse dilations across routes in a scene. If
    `free` is given, that single map is used and adaptivity is skipped (back-compat)."""
    shape = tall.shape
    if free is not None:
        return _plan_on(free, start_xy, goal_xy, extent, shape) or [np.asarray(goal_xy, float)]
    if levels is None:
        levels = build_levels(tall, extent)
    sxy = np.asarray(start_xy, float); gxy = np.asarray(goal_xy, float)
    for infl, fr, lbl in levels:                       # widest clearance first
        s = _nearest_free(fr, world_to_px(sxy, extent, shape))
        g = _nearest_free(fr, world_to_px(gxy, extent, shape))
        if lbl[s] != 0 and lbl[s] == lbl[g]:           # same connected component => a path exists
            wps = _plan_on(fr, sxy, gxy, extent, shape)
            if wps is not None:
                return wps
    # no clearance level connected them; try the finest free map directly, else straight line
    wps = _plan_on(levels[-1][1], sxy, gxy, extent, shape)
    return wps if wps is not None else [gxy]
