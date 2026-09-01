#!/usr/bin/env python3
"""BLOCKER CHECK for the geometry-grounded tokenizer (SceMoS port, IN_FLIGHT "Next direction").

The open question the docs say to answer BEFORE any VQ-VAE retrain: does HUMANISE actually
give us usable per-frame surface geometry to build SceMoS-style local heightmaps? SceMoS feeds
its VQ decoder a local heightmap (+-0.6 m in the body frame, 32x32) recomputed each step, with
a contact loss, and gets contact-correct interaction. Ours is scene-blind. This tests whether
the input even exists at usable resolution on our data -- a cheap, decisive gate (CLAUDE.md
S5: test cheaply, pivot early), NOT a retrain.

Everything needed is already in hand:
  - compute_track2(rec) places the 22 joints into the ScanNet WORLD frame (same frame as the
    mesh -- it subtracts scene_translation, floor-overlay-validated 150/150, RESULTS S1), so
    pelvis world (x, y, z) and yaw are known per frame.
  - bev_render._load_scene_mesh(scene) loads that mesh; its vertices are the surface geometry.
A local body-frame heightmap under the root is then just: sample max mesh-surface height over a
+-0.6 m grid rotated to the body yaw. No model, no GPU, no 263 conversion (avoids the MDM dep).

THE ORACLE (what makes this a real measurement, not a vibe): at a sit/lie CONTACT frame the
heightmap directly UNDER the body must read a raised support (seat ~0.45 m, bed height), while
under a WALKING body it must read ~floor (~0). If sit/lie show a raised support and walk shows
floor, the signal is real and the port is viable. If the seat reads as floor (mesh missing the
furniture, wrong frame, or too-sparse verts), the direction is BLOCKED -- surface that here.

Outputs: a per-action table (support-under-body vs pelvis height, cell fill-rate) and montage
PNGs of example heightmaps with the body-center marked, under --out.
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from humanise_join import build_flat_join, get_record, compute_track2, J_PELVIS  # noqa: E402
import bev_render  # noqa: E402

GRID_N = 32          # SceMoS grid resolution
HALF_EXTENT = 0.6    # +-0.6 m body frame, SceMoS
QUERY_R = 0.06       # radius to gather surface verts per cell (cell size = 1.2/32 = 0.0375 m)


def local_heightmap(mesh_kdt, verts_z, floor_z, center_xy, yaw, n=GRID_N, half=HALF_EXTENT,
                    query_r=QUERY_R):
    """Body-frame heightmap under the root. u=forward(yaw), v=left(yaw+90). Each cell = max
    surface height (above floor) among verts within query_r; empty cells -> nearest vert (dense,
    SceMoS-style). Returns (hm (n,n) height-above-floor, fill_frac raw within-radius fill)."""
    ax = np.linspace(-half, half, n)
    uu, vv = np.meshgrid(ax, ax, indexing="ij")          # uu=forward, vv=left
    c, s = np.cos(yaw), np.sin(yaw)
    wx = center_xy[0] + uu * c + vv * (-s)
    wy = center_xy[1] + uu * s + vv * (c)
    pts = np.stack([wx.ravel(), wy.ravel()], axis=1)     # (n*n, 2)

    hm = np.full(pts.shape[0], np.nan, dtype=np.float64)
    neigh = mesh_kdt.query_ball_point(pts, r=query_r)
    for i, idxs in enumerate(neigh):
        if idxs:
            hm[i] = verts_z[idxs].max()
    fill_frac = np.isfinite(hm).mean()
    # densify empties with nearest surface vert so the map is complete (SceMoS heightmap is dense)
    empty = ~np.isfinite(hm)
    if empty.any():
        _, nn = mesh_kdt.query(pts[empty], k=1)
        hm[empty] = verts_z[nn]
    hm = hm.reshape(n, n) - floor_z
    return hm, fill_frac


def pick(action, n, seed, need_interact=True):
    """Indices for `action`; for sit/lie keep only clips whose GT actually goes low (interacts),
    judged by pelvis world height at the anchor frame -- no 263 path needed."""
    flat = build_flat_join()
    idxs = [i for i, p in enumerate(flat) if p["action"] == action]
    rng = np.random.RandomState(seed)
    rng.shuffle(idxs)
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="clips per action to score")
    ap.add_argument("--actions", nargs="+", default=["sit", "lie", "walk", "stand up"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/wander_data/scene_tokenizer_probe"))
    ap.add_argument("--montage", type=int, default=6, help="example heightmaps per action to render")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mesh_cache = {}   # scene -> (kdtree, verts_z, floor_z)

    def get_scene(scene):
        if scene not in mesh_cache:
            if len(mesh_cache) > 40:
                mesh_cache.clear()
            try:
                mesh = bev_render._load_scene_mesh(scene)
            except Exception as e:
                mesh_cache[scene] = None
                return None
            v = np.asarray(mesh.vertices, dtype=np.float64)
            mesh_cache[scene] = (cKDTree(v[:, :2]), v[:, 2], v[:, 2].min())
        return mesh_cache[scene]

    summary = {}
    montage = {}   # action -> list of (hm, pelvis_h, support_h)
    for action in args.actions:
        idxs = pick(action, args.n, args.seed)
        pelvis_hs, support_hs, fills, deltas = [], [], [], []
        used = 0
        for idx in idxs:
            if used >= args.n:
                break
            rec = get_record(int(idx))
            sc = get_scene(rec.scene)
            if sc is None:
                continue
            kdt, verts_z, floor_z = sc
            jw, xy, yaw, _ = compute_track2(rec)
            if jw.shape[0] < 4:
                continue
            anchor = rec.anchor_frame   # sit/lie/walk -> -1 (final), stand up -> 0
            pelvis_h = float(jw[anchor, J_PELVIS, 2] - floor_z)
            # interaction filter: sit/lie must actually be low at the contact frame
            if action == "sit" and pelvis_h > 0.75:
                continue
            if action == "lie" and pelvis_h > 0.5:
                continue
            hm, fill = local_heightmap(kdt, verts_z, floor_z, xy[anchor], float(yaw[anchor]))
            # support directly under the body = center 3x3 cells (max), the surface it rests on
            cc = GRID_N // 2
            support_h = float(np.nanmax(hm[cc - 1:cc + 2, cc - 1:cc + 2]))
            pelvis_hs.append(pelvis_h)
            support_hs.append(support_h)
            fills.append(fill)
            deltas.append(pelvis_h - support_h)   # body clearance above the surface it's on
            if len(montage.setdefault(action, [])) < args.montage:
                montage[action].append((hm, pelvis_h, support_h, rec.scene, idx))
            used += 1
        summary[action] = dict(
            n=used,
            pelvis_h=float(np.mean(pelvis_hs)) if pelvis_hs else float("nan"),
            support_h=float(np.mean(support_hs)) if support_hs else float("nan"),
            clearance=float(np.mean(deltas)) if deltas else float("nan"),
            fill=float(np.mean(fills)) if fills else float("nan"),
        )

    # ---- report ----
    print(f"\n=== local heightmap probe (+-{HALF_EXTENT} m, {GRID_N}x{GRID_N}, "
          f"query_r={QUERY_R} m) ===")
    print(f"{'action':9s} {'n':>3s} {'pelvis_h':>9s} {'support_h':>10s} {'clearance':>10s} "
          f"{'fill%':>6s}")
    for a in args.actions:
        s = summary.get(a)
        if not s:
            continue
        print(f"{a:9s} {s['n']:3d} {s['pelvis_h']:9.3f} {s['support_h']:10.3f} "
              f"{s['clearance']:10.3f} {100*s['fill']:6.0f}")
    print("\nread: support_h = surface height under the body (above floor). For sit/lie it must")
    print("be RAISED (seat/bed) not ~0; for walk it must be ~floor. clearance = pelvis - support")
    print("(seat-to-pelvis gap when seated; ~pelvis height when walking on floor). fill% = cells")
    print("with a real surface vertex within query_r (low => grid finer than the mesh).")

    # ---- montage PNGs ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for a, items in montage.items():
            if not items:
                continue
            k = len(items)
            fig, axes = plt.subplots(1, k, figsize=(2.4 * k, 2.8))
            if k == 1:
                axes = [axes]
            vmax = max(np.nanmax(hm) for hm, *_ in items)
            for ax, (hm, ph, sh, scene, idx) in zip(axes, items):
                im = ax.imshow(hm.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                               extent=[-HALF_EXTENT, HALF_EXTENT, -HALF_EXTENT, HALF_EXTENT])
                ax.plot(0, 0, "rx", ms=8, mew=2)     # body center
                ax.set_title(f"{scene}\n#{idx} p={ph:.2f} s={sh:.2f}", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=axes, fraction=0.02, label="height above floor (m)")
            fig.suptitle(f"{a}: local body-frame heightmap (x = body center, forward = +x)",
                         fontsize=9)
            p = os.path.join(args.out, f"heightmap_{a.replace(' ', '_')}.png")
            fig.savefig(p, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {p}")
    except Exception as e:
        print(f"(montage skipped: {e})")


if __name__ == "__main__":
    main()
