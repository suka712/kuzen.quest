"""Local body-frame surface heightmap under the root, for the SceMoS-style geometry-grounded
tokenizer (IN_FLIGHT "Next direction"). ONE definition, shared by the extractor (training data)
and inference -- so the heightmap the decoder sees at inference is sampled the identical way it
was trained.

Geometry (matches SceMoS: +-0.6 m body frame, 32x32, recomputed per frame):
  - Grid axes: u = forward (body yaw), v = left (yaw + 90deg). Cell (i,j) world point =
    center + u_i * [cos yaw, sin yaw] + v_j * [-sin yaw, cos yaw].
  - Value = max scene-surface height among mesh verts within QUERY_R of the cell; empty cells ->
    nearest surface vert (dense map).
  - CEILING CUT: verts higher than CEILING_M above the SCENE floor are dropped before sampling,
    so a ceiling / overhead beam / high shelf beside the target cannot dominate a column (the
    probe saw one seat read 2.5 m from exactly this). Walls/furniture below survive as obstacle
    signal.

VERTICAL REFERENCE -- LOAD-BEARING (learned 2026-08-28 the hard way, RESULTS §7 floor trap):
  local_heightmap returns ABSOLUTE surface height (scene-mesh Z). The stored/served heightmap must
  be re-referenced to the CLIP FLOOR via to_clip_frame(), because the 263 the decoder reconstructs
  is clip-floor-relative (process_file sets floor = min joint height over the clip). For a
  lie-on-bed clip the clip floor IS the bed (~0.8 m up); leaving the heightmap scene-referenced put
  the whole body ~0.8 m "under" the surface (GT penetration 400-770 mm). Clip-floor referencing
  (surface_z - clip_floor_world_z) drops GT penetration to ~0 and preserves the signal that
  matters: a chair seat reads RAISED above the ground the feet stand on, while a bed the whole body
  rests on becomes the local floor -- consistent with the clip-floor-relative motion.

FRAME alignment (verified 2026-08-28): 263 feature index t = raw frame t (process_file drops the
LAST frame; T263 = Traw - 1), so heightmap[t] is sampled at compute_track2's world (x, y, yaw)[t],
offset 0.
"""
import numpy as np
from scipy.spatial import cKDTree

GRID_N = 32
HALF_EXTENT = 0.6      # +-0.6 m, SceMoS
QUERY_R = 0.06         # cell size = 1.2/32 = 0.0375 m
CEILING_M = 2.0        # drop verts above this height above the SCENE floor
HM_LOW, HM_HIGH = -1.0, 2.0   # clip range for the clip-floor-referenced heightmap


def build_scene_kdtree(vertices):
    """(N,3) Z-up world verts -> (kdtree over XY of below-ceiling verts, their ABSOLUTE Z,
    scene_floor_z). Reused across all frames of every clip in a scene."""
    v = np.asarray(vertices, dtype=np.float64)
    floor_z = float(v[:, 2].min())
    keep = (v[:, 2] - floor_z) <= CEILING_M
    v = v[keep]
    return cKDTree(v[:, :2]), v[:, 2].copy(), floor_z


def local_heightmap(kdt, verts_z, center_xy, yaw, n=GRID_N, half=HALF_EXTENT, query_r=QUERY_R):
    """Body-frame heightmap under the root. Returns (hm (n,n) ABSOLUTE surface Z float32,
    fill_frac). Re-reference to the clip floor with to_clip_frame(). kdt/verts_z from
    build_scene_kdtree."""
    ax = np.linspace(-half, half, n)
    uu, vv = np.meshgrid(ax, ax, indexing="ij")          # uu = forward, vv = left
    c, s = np.cos(yaw), np.sin(yaw)
    wx = center_xy[0] + uu * c + vv * (-s)
    wy = center_xy[1] + uu * s + vv * c
    pts = np.stack([wx.ravel(), wy.ravel()], axis=1)     # (n*n, 2)

    hm = np.full(pts.shape[0], np.nan, dtype=np.float64)
    for i, idxs in enumerate(kdt.query_ball_point(pts, r=query_r)):
        if idxs:
            hm[i] = verts_z[idxs].max()
    fill_frac = float(np.isfinite(hm).mean())
    empty = ~np.isfinite(hm)
    if empty.any():
        _, nn = kdt.query(pts[empty], k=1)
        hm[empty] = verts_z[nn]
    return hm.reshape(n, n).astype(np.float32), fill_frac


def to_clip_frame(hm_abs, clip_floor_z, low=HM_LOW, high=HM_HIGH):
    """Absolute-Z heightmap -> clip-floor-referenced, clipped to [low, high]. clip_floor_z =
    min joint world Z over the clip's used frames (matches process_file's floor)."""
    return np.clip(hm_abs - clip_floor_z, low, high).astype(np.float32)


def flat_floor_heightmap(n=GRID_N):
    """Heightmap for a scene-less clip (H3D locomotion): flat floor at the clip floor (0)."""
    return np.zeros((n, n), dtype=np.float32)
