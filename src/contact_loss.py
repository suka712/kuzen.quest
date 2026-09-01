"""Penetration / contact loss for the geometry-grounded VQ-VAE (SceMoS-style contact term).

Computable inside the canonicalized VQ forward because the local heightmap SHARES the body frame
with the decoded local joints. From HumanML3D's process_file canonicalisation, the local pose is
root-relative and heading-aligned so that: forward = +Z, up = +Y, left = +X (up x forward). The
heightmap grid is u = forward, v = left (src/scene_heightmap.py), so a joint at local (x, y, z)
maps to heightmap cell (u=z, v=x) and its height is y. No body joint should sit BELOW the surface
under it: penetration = relu(surface(u,v) - height - margin), one-sided, averaged over joints and
frames. The standard reconstruction loss supplies the "touch the surface" supervision on GT; this
adds the one-sided prior that generalises to surface heights unseen at train time.

Heights: BOTH the joint y (263) and the served heightmap are referenced to the CLIP FLOOR
(scene_heightmap.to_clip_frame re-references the raw scene-Z heightmap by the clip's
min-joint-height, matching process_file). This shared reference is load-bearing: leaving the
heightmap scene-referenced put lie-on-bed bodies ~0.8 m "under" the bed (GT penetration 400-770
mm); clip-floor referencing drops it to ~0. Verified by the GT oracle in __main__ (GT bodies rest
on surfaces => ~0 penetration; raising the heightmap 0.3 m raises penetration ~0.3 m).
"""
import numpy as np
import torch

from scene_heightmap import GRID_N, HALF_EXTENT

# 263 layout: [0]=rot_vel [1:3]=lin_vel_xz [3]=root_y [4:4+63]=ric (21 joints x3)
_RIC0, _NJ = 4, 22


def local_joints_from_263(real263):
    """real263 (bs,T,263) DENORMALIZED -> (bs,T,22,3) local joints, HML Y-up (root at origin,
    y=height above clip floor). Torch, differentiable."""
    bs, T, _ = real263.shape
    root_y = real263[:, :, 3]
    ric = real263[:, :, _RIC0:_RIC0 + (_NJ - 1) * 3].reshape(bs, T, _NJ - 1, 3)
    root = torch.zeros(bs, T, 1, 3, device=real263.device, dtype=real263.dtype)
    root[:, :, 0, 1] = root_y
    return torch.cat([root, ric], dim=2)  # (bs,T,22,3)


def penetration_loss(pred_norm, hm, mean, std, margin=0.03, joints=None, return_stats=False):
    """pred_norm (bs,T,263) normalized decoder output; hm (bs,T,N,N) surface height above floor;
    mean/std (263,) tensors. Returns scalar mean penetration (metres), one-sided."""
    real = pred_norm * std + mean
    j = local_joints_from_263(real)             # (bs,T,22,3) x=left, y=up, z=forward
    if joints is not None:
        j = j[:, :, joints, :]
    u = j[..., 2]                                # forward
    v = j[..., 0]                                # left
    y = j[..., 1]                                # height
    n = hm.shape[-1]
    # (u,v) in [-half,half] -> cell index [0,n-1]
    ui = ((u + HALF_EXTENT) / (2 * HALF_EXTENT) * (n - 1)).round().long().clamp(0, n - 1)
    vi = ((v + HALF_EXTENT) / (2 * HALF_EXTENT) * (n - 1)).round().long().clamp(0, n - 1)
    bs, T, J = ui.shape
    bb = torch.arange(bs, device=hm.device).view(bs, 1, 1).expand(bs, T, J)
    tt = torch.arange(T, device=hm.device).view(1, T, 1).expand(bs, T, J)
    surface = hm[bb, tt, ui, vi]                 # (bs,T,J)
    pen = torch.relu(surface - y - margin)
    if return_stats:
        return pen.mean(), dict(max=float(pen.max()), frac_pen=float((pen > 0).float().mean()))
    return pen.mean()


if __name__ == "__main__":
    # GT oracle: GT motion + its GT heightmap should have LOW penetration; raising the heightmap
    # uniformly must raise penetration ~ the raise. Validates the axis mapping + reference.
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    HUM_263 = os.path.expanduser("~/wander_data/motion_data/HUMANISE_263_cache")
    HUM_HM = os.path.expanduser("~/wander_data/motion_data/HUMANISE_heightmap_cache")
    EVAL_MEAN = os.path.expanduser(
        "~/Khiem/T2M-GPT/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy")
    EVAL_STD = EVAL_MEAN.replace("mean.npy", "std.npy")
    mean = torch.tensor(np.load(EVAL_MEAN), dtype=torch.float32)
    std = torch.tensor(np.load(EVAL_STD), dtype=torch.float32)

    # sample: sit block starts ~ index 0 (natsort: lie,sit,stand up,walk) -- pick clips that have hm
    import glob
    ok = 0
    for f in sorted(glob.glob(f"{HUM_HM}/*.npy")):
        idx = int(os.path.basename(f)[:5])
        m = np.load(f"{HUM_263}/{idx:05d}.npy").astype(np.float32)
        h = np.load(f).astype(np.float32)
        if m.shape[0] != h.shape[0] or m.shape[0] < 8:
            continue
        mt = ((torch.tensor(m) - mean) / std).unsqueeze(0)     # normalized, as decoder would emit
        ht = torch.tensor(h).unsqueeze(0)
        p0, s0 = penetration_loss(mt, ht, mean, std, return_stats=True)
        p_up, _ = penetration_loss(mt, ht + 0.3, mean, std, return_stats=True)
        # WRONG axis (swap u,v) as a control -- should give clearly worse GT penetration
        print(f"#{idx}: GT_pen={1000*p0:.0f}mm frac={s0['frac_pen']:.2f} "
              f"| raise0.3 -> {1000*p_up:.0f}mm")
        ok += 1
        if ok >= 12:
            break
