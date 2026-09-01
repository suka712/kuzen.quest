#!/usr/bin/env python3
"""Step 12 figure: greedy decoding COLLIDES, collision-guided decoding AVOIDS -- same scene, same
waypoints, same start. Scans scenes (same seed/order as collision_guided.py) and renders the one
with the largest greedy->guided collision improvement, so the figure is a REAL measured case, not a
hand-picked toy. Left panel = greedy path; right panel = guided_seg path; both over the 0.9 m
tall-obstacle map (red) with the walkable occupancy (gray) beneath. Path segments on an obstacle are
drawn thick red so a collision is visible at a glance.
"""
import argparse
import os
import sys

import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import load_model  # noqa: E402
from demo_rollout import sample_waypoints  # noqa: E402
from collision_guided import (run_chain, path_collision, straight_line_collision,  # noqa: E402
                              T2M, HUMANISE, BEV, TALL, DEV)


def on_obstacle(path_xy, tall, extent):
    xmin, xmax, ymin, ymax = extent
    H, W = tall.shape
    c = np.clip(((path_xy[:, 0] - xmin) / (xmax - xmin) * W).astype(int), 0, W - 1)
    r = np.clip(((ymax - path_xy[:, 1]) / (ymax - ymin) * H).astype(int), 0, H - 1)
    return tall[r, c] > 0.5, c, r


def draw_panel(ax, occ, tall, extent, segs, wps, start_xy, title):
    H, W = tall.shape
    rgb = np.ones((H, W, 3), np.float32)
    rgb[occ > 0.5] = (0.75, 0.75, 0.78)          # low furniture / walkable-blocked (gray)
    rgb[tall > 0.5] = (0.95, 0.55, 0.5)          # 0.9 m tall obstacles (salmon)
    ax.imshow(rgb, origin="upper")
    path = np.concatenate([s["world"][:, 0, :2] for s in segs])
    hit, c, r = on_obstacle(path, tall, extent)
    ax.plot(c, r, "-", color="#1a7f37", lw=2.2, zorder=3)      # the route
    ax.plot(c[hit], r[hit], ".", color="#d1242f", ms=6, zorder=4)  # collision samples
    xmin, xmax, ymin, ymax = extent

    def w2px(p):
        return (np.clip((p[0] - xmin) / (xmax - xmin) * W, 0, W - 1),
                np.clip((ymax - p[1]) / (ymax - ymin) * H, 0, H - 1))
    for wp in wps:
        gx, gy = w2px(wp)
        ax.plot([gx], [gy], "*", color="#0969da", ms=11, zorder=5)
    sx, sy = w2px(start_xy)
    ax.plot([sx], [sy], "o", color="#111", ms=9, zorder=5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-scan", type=int, default=20, help="scenes to scan for the best contrast")
    ap.add_argument("--n-segments", type=int, default=6)
    ap.add_argument("--min-step", type=float, default=0.6)
    ap.add_argument("--max-step", type=float, default=1.2)
    ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--coll-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default=None, help="force a specific scene id (e.g. scene0001_00)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)

    flat = build_flat_join()
    walk_idx = [i for i, p in enumerate(flat) if p["action"] == "walk"]
    rng = np.random.RandomState(args.seed); rng.shuffle(walk_idx)

    best = None
    scanned = 0
    for idx in walk_idx:
        if scanned >= args.n_scan and best is not None:
            break
        rec = get_record(int(idx))
        if args.scene and rec.scene != args.scene:
            continue
        fb, ft = os.path.join(BEV, f"{rec.scene}.npz"), os.path.join(TALL, f"{rec.scene}.npz")
        if not (os.path.exists(fb) and os.path.exists(ft)):
            continue
        zb, zt = np.load(fb), np.load(ft)
        occ, extent = zb["occ"].astype(np.float32), zb["extent"]
        tall = zt["occ"].astype(np.float32)
        cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
        try:
            d0, *_ = mf.humanise_positions_to_263(cm)
        except Exception:
            continue
        if d0.shape[0] < 8:
            continue
        _, xy, _, sincos = compute_track2(rec)
        start_pose = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
        prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel()
        wps = sample_waypoints(occ, extent, start_pose[:2], args.n_segments, args.min_step,
                               rng, max_step=args.max_step)
        if wps is None:
            continue
        texts = ["walk to the target"] * args.n_segments
        g = run_chain(trans, net, cmodel, mean, std, ns, texts, wps, start_pose, prefix,
                      occ, extent, tall, True, "greedy", 1, 0.0, rng)
        gd = run_chain(trans, net, cmodel, mean, std, ns, texts, wps, start_pose, prefix,
                       occ, extent, tall, True, "guided_seg", args.n_cand, args.coll_weight, rng)
        if len(g) < args.n_segments or len(gd) < args.n_segments:
            continue
        gc = path_collision(np.concatenate([s["world"][:, 0, :2] for s in g]), tall, extent)
        dc = path_collision(np.concatenate([s["world"][:, 0, :2] for s in gd]), tall, extent)
        scanned += 1
        print(f"  scan {scanned} {rec.scene} greedy={gc*100:.1f}% guided={dc*100:.1f}% "
              f"(improve {(gc-dc)*100:.1f}pp)", flush=True)
        if best is None or (gc - dc) > best["impr"]:
            best = dict(impr=gc - dc, rec=rec, occ=occ, tall=tall, extent=extent,
                        g=g, gd=gd, gc=gc, dc=dc, wps=wps, start=start_pose[:2].copy())
        if args.scene:
            break

    if best is None:
        print("nothing rendered"); return
    line = straight_line_collision(best["start"], best["wps"], best["tall"], best["extent"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.4))
    draw_panel(ax[0], best["occ"], best["tall"], best["extent"], best["g"], best["wps"],
               best["start"], f"greedy decoding — collision {best['gc']*100:.1f}%")
    draw_panel(ax[1], best["occ"], best["tall"], best["extent"], best["gd"], best["wps"],
               best["start"], f"collision-guided decoding — collision {best['dc']*100:.1f}%")
    fig.suptitle(f"{best['rec'].scene}: guided decoding steers around tall obstacles "
                 f"(salmon)  ·  straight-line control {line*100:.1f}%", fontsize=12)
    fig.tight_layout()
    p = os.path.join(args.out, f"cg_compare_{best['rec'].scene}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nrendered {p}\n  greedy {best['gc']*100:.1f}%  guided {best['dc']*100:.1f}%  "
          f"line {line*100:.1f}%")


if __name__ == "__main__":
    main()
