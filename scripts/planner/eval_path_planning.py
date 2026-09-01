#!/usr/bin/env python3
"""Reproduce the demo's wall penetration and validate the global path planner (src/grid_planner).

For each scene, sample free-floor START points and route each to every furniture ANCHOR two ways:
  straight  the CURRENT demo behaviour (expand_plan straight-line hops) -- start->goal in a line.
  planned   grid_planner.plan_path -- A* around walls on the inflated 0.9 m tall raster.
Both are scored as the fraction of the DENSE root path that lands on the tall (wall) raster
(collision_guided.straight_line_collision, the same metric the demo prints).

Expectation: straight collision > 0 on routes that cross a wall (the reported bug); planned ~0.
This is the oracle-style control -- if planned is NOT ~0, the planner is broken, not the model.
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining", "scripts/planner"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

from scene_anchors import load_scene_maps, detect_anchors  # noqa: E402
from collision_guided import straight_line_collision  # noqa: E402
from grid_planner import plan_path, build_levels  # noqa: E402
from demo_rollout import sample_waypoints  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=["scene0000_00", "scene0151_00", "scene0641_00", "scene0008_00"])
    ap.add_argument("--starts", type=int, default=8, help="free-floor starts per scene")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    tot_str, tot_pln, n_pairs, n_wall = [], [], 0, 0
    for sc in args.scenes:
        try:
            rgb, occ, tall, extent = load_scene_maps(sc)
        except Exception as e:
            print(f"{sc}: load err {e}"); continue
        occ = occ.astype(np.float32); tall = tall.astype(np.float32)
        anchors = detect_anchors(occ.astype(bool), tall.astype(bool), extent)
        if not anchors:
            print(f"{sc}: no anchors"); continue
        levels = build_levels(tall, extent)  # adaptive clearance, reused across all routes
        # sample free starts far from furniture (real walk-ups)
        starts = []
        for _ in range(args.starts * 6):
            wp = sample_waypoints(occ, extent, np.array([extent[0], extent[2]]), 1,
                                  min_step=0.5, rng=rng, max_step=99.0)
            if wp and all(np.linalg.norm(np.asarray(wp[0]) - a["xy"]) > 1.2 for a in anchors):
                starts.append(np.asarray(wp[0], float))
            if len(starts) >= args.starts:
                break
        s_str, s_pln, s_wall = [], [], 0
        for st in starts:
            for a in anchors:
                goal = a["xy"].astype(float)
                cs = straight_line_collision(st, [goal], tall, extent) * 100
                wps = plan_path(st, goal, tall, extent, levels=levels)
                cp = straight_line_collision(st, wps, tall, extent) * 100
                s_str.append(cs); s_pln.append(cp); n_pairs += 1
                if cs > 1.0:
                    s_wall += 1; n_wall += 1
        tot_str += s_str; tot_pln += s_pln
        print(f"{sc}: {len(starts)} starts x {len(anchors)} anchors = {len(s_str)} routes | "
              f"straight coll mean {np.mean(s_str):.1f}% (max {np.max(s_str):.1f}%), "
              f"{s_wall} cross a wall (>1%) | planned coll mean {np.mean(s_pln):.2f}% "
              f"(max {np.max(s_pln):.2f}%)")

    if n_pairs:
        ts, tp = np.array(tot_str), np.array(tot_pln)
        print(f"\n=== ALL {n_pairs} routes ===")
        print(f"straight (current demo): mean {ts.mean():.1f}%  max {ts.max():.1f}%  "
              f"| {n_wall} routes cross a wall (>1%)")
        print(f"planned  (A* detour)   : mean {tp.mean():.2f}%  max {tp.max():.2f}%")
        wall = ts > 1.0
        if wall.any():
            print(f"on the {wall.sum()} WALL-CROSSING routes: straight {ts[wall].mean():.1f}% "
                  f"-> planned {tp[wall].mean():.2f}%  (this is the demo bug, fixed)")


if __name__ == "__main__":
    main()
