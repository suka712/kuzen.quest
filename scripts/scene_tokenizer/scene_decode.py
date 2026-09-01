"""Inference-side heightmap decoding for the geometry-grounded tokenizer (Stage 5).

The decoder now takes (tokens, local heightmap), but the heightmap depends on WHERE each frame
lands in the world, which depends on the decode -> a circularity. Resolved exactly as sketched in
IN_FLIGHT: scene-blind first-pass decode (flat-floor heightmap) -> SE(2)-place -> sample the real
per-frame heightmaps along that placed trajectory -> re-decode. One or two refinement iterations.

Vertical reference at inference = the SCENE mesh floor (height above ground), which matches the
canonical motion's own floor (feet ~0) for ground-based interaction -- a person who walks to a
chair on the floor and sits. (A lie-on-a-high-bed segment, where the whole body leaves the ground,
would want the bed as the reference; the walk->sit->stand demo is ground-based, so scene-floor is
correct there. Noted as a limitation for lie chains.)

Frozen encoder+quantizer => tokens are unchanged, so the step-11 transformer feeds this decoder
directly; only forward_decoder gains the heightmap argument.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import bev_render  # noqa: E402
from scene_heightmap import (build_scene_kdtree, local_heightmap, to_clip_frame,  # noqa: E402
                             flat_floor_heightmap, GRID_N)
from se2_utils import se2_place_full_body  # noqa: E402
from humanise_join import J_LHIP, J_RHIP, J_LSHOULDER, J_RSHOULDER  # noqa: E402


def build_scene_context(scene_id):
    """(kdt, verts_z_abs, scene_floor) for a ScanNet scene, reused across all segments/frames."""
    mesh = bev_render._load_scene_mesh(scene_id)
    kdt, verts_z, floor_z = build_scene_kdtree(mesh.vertices)
    return kdt, verts_z, floor_z


def _yaw_from_pose(p22):
    across = (p22[J_RHIP, :2] - p22[J_LHIP, :2]) + (p22[J_RSHOULDER, :2] - p22[J_LSHOULDER, :2])
    n = np.linalg.norm(across)
    across = across / (n if n > 1e-8 else 1e-8)
    return float(np.arctan2(across[0], -across[1]))  # = atan2(fwd_y, fwd_x), fwd=(-across_y,across_x)


def local_ground(scene_ctx, xy, r=0.2):
    """The floor height under a standing point = the LOW surface within r of xy (min over nearby
    verts). This is the inference analog of training's clip_floor (min joint z = feet on floor);
    the global mesh min-z is wrong (a sub-floor scan-noise vertex elsewhere inflates every hm)."""
    kdt, verts_z, scene_floor = scene_ctx
    idx = kdt.query_ball_point(np.asarray(xy, float), r=r)
    if not idx:
        _, nn = kdt.query(np.asarray(xy, float), k=8)
        idx = np.atleast_1d(nn)
    return float(np.percentile(verts_z[idx], 5))   # 5th pct = floor, robust to a few high verts


def sample_track_heightmaps(scene_ctx, world_traj, floor_ref):
    """world_traj (T,22,3) Z-up placed body -> (T,32,32) heightmaps sampled under the pelvis,
    oriented to the per-frame body facing, referenced to floor_ref (the local ground)."""
    kdt, verts_z, _ = scene_ctx
    T = world_traj.shape[0]
    hm = np.empty((T, GRID_N, GRID_N), dtype=np.float32)
    for t in range(T):
        xy = world_traj[t, 0, :2]
        yaw = _yaw_from_pose(world_traj[t])
        h_abs, _ = local_heightmap(kdt, verts_z, xy, yaw)
        hm[t] = to_clip_frame(h_abs, floor_ref)
    return hm


def decode_with_heightmap(scene_net, tok, pose, scene_ctx, mf_module, std, mean, n_iters=2):
    """Decode `tok` conditioned on the scene, resolving the heightmap circularity.

    scene_net: SceneVQVAE (forward_decoder(tok, hm)). pose: SE(2) start (x,y,sin,cos). Returns the
    final canonicalized motion (T,263) numpy (denormalized), same type rollout expects from
    net.forward_decoder(tok)[0]*std+mean."""
    import torch
    floor_ref = local_ground(scene_ctx, np.asarray(pose[:2], float))  # ground where the body starts
    # Pass 1: flat-floor prior (we don't know the surface until we place the body).
    Tguess = tok.shape[-1] * 4
    hm = flat_floor_heightmap()[None].repeat(Tguess, 0)  # (T,32,32); T corrected below
    for it in range(n_iters + 1):
        ht = torch.from_numpy(hm).float().unsqueeze(0).to(tok.device)
        motion = scene_net.forward_decoder(tok, ht)[0].cpu().numpy() * std + mean
        if it == n_iters:
            return motion.astype(np.float32)
        world = se2_place_full_body(motion.astype(np.float32), pose, mf_module)  # (T,22,3)
        hm = sample_track_heightmaps(scene_ctx, world, floor_ref)  # aligned to motion length
    return motion.astype(np.float32)
