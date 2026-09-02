#!/usr/bin/env python3
"""Top-down figure: the demo's OLD straight-line route walks through a wall; the grid planner routes
around it. Overlays both routes on the scene BEV with the 0.9 m tall (wall) raster highlighted."""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/planner", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

from scene_anchors import load_scene_maps  # noqa: E402
from grid_planner import build_levels, plan_path, furniture_obstacle  # noqa: E402
from collision_guided import straight_line_collision  # noqa: E402


def dense(a, b, step=0.03):
    n = max(2, int(np.linalg.norm(b - a) / step))
    return np.linspace(a, b, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0151_00")
    ap.add_argument("--start", default="3.88,8.79")
    ap.add_argument("--goal", default="3.76,4.00")
    ap.add_argument("--out", default=os.path.expanduser("~/wander_data/step16_demo/path_planning_fig.png"))
    args = ap.parse_args()
    st = np.array([float(v) for v in args.start.split(",")])
    goal = np.array([float(v) for v in args.goal.split(",")])
    rgb, occ, tall, extent = load_scene_maps(args.scene)
    tall = tall.astype(float); occ = occ.astype(float)
    xmin, xmax, ymin, ymax = extent
    # obstacle map that avoids walls + all furniture EXCEPT the target couch
    furn = furniture_obstacle(tall, occ, extent, target_xy=goal).astype(float)
    furn_levels = build_levels(furn, extent)
    wps = plan_path(st, goal, furn, extent, levels=furn_levels)
    planned = np.vstack([st, np.asarray(wps, float)])
    # score the STRAIGHT route and the planned route against the furniture map (what "through the chair" means)
    cs = straight_line_collision(st, [goal], furn, extent) * 100
    cp = straight_line_collision(st, wps, furn, extent) * 100

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb, extent=[xmin, xmax, ymin, ymax], origin="upper")
    low = np.ma.masked_where(~((occ > 0.5) & (tall < 0.5)), occ)   # low furniture
    wall = np.ma.masked_where(tall < 0.5, tall)                    # walls
    ax.imshow(low, extent=[xmin, xmax, ymin, ymax], origin="upper", cmap="cool",
              alpha=0.5, vmin=0, vmax=1)
    ax.imshow(wall, extent=[xmin, xmax, ymin, ymax], origin="upper", cmap="autumn",
              alpha=0.55, vmin=0, vmax=1)
    ax.plot([st[0], goal[0]], [st[1], goal[1]], "--", color="red", lw=2.5,
            label=f"old: straight ({cs:.0f}% through furniture)")
    ax.plot(planned[:, 0], planned[:, 1], "-", color="lime", lw=2.5,
            label=f"furniture-aware ({cp:.1f}%)")
    ax.plot(planned[:, 0], planned[:, 1], "o", color="lime", ms=4)
    ax.scatter([st[0]], [st[1]], c="dodgerblue", s=140, zorder=5, edgecolors="k", label="start")
    ax.scatter([goal[0]], [goal[1]], c="yellow", s=180, marker="*", zorder=5, edgecolors="k", label="couch (goal)")
    ax.set_title(f"{args.scene}: furniture-aware routing (magenta=furniture, yellow=walls)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"straight {cs:.1f}%  planned {cp:.2f}%  ({len(wps)} waypoints) -> {args.out}")


if __name__ == "__main__":
    main()
