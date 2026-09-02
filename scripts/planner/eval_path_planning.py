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
from grid_planner import plan_path, build_levels, furniture_obstacle  # noqa: E402
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
        walls_levels = build_levels(tall, extent)   # walls-only planner (the previous fix)
        # sample free starts far from furniture (real walk-ups)
        starts = []
        for _ in range(args.starts * 6):
            wp = sample_waypoints(occ, extent, np.array([extent[0], extent[2]]), 1,
                                  min_step=0.5, rng=rng, max_step=99.0)
            if wp and all(np.linalg.norm(np.asarray(wp[0]) - a["xy"]) > 1.2 for a in anchors):
                starts.append(np.asarray(wp[0], float))
            if len(starts) >= args.starts:
                break
        s_str, s_wallp, s_furn = [], [], []
        for st in starts:
            for a in anchors:
                goal = a["xy"].astype(float)
                # score every routing against the FURNITURE map (walls + non-target furniture):
                # this is what "walks through the chair" actually means.
                furn = furniture_obstacle(tall, occ, extent, target_xy=goal).astype(np.float32)
                furn_levels = build_levels(furn, extent)
                cs = straight_line_collision(st, [goal], furn, extent) * 100         # straight line
                wl = straight_line_collision(st, plan_path(st, goal, tall, extent, levels=walls_levels),
                                             furn, extent) * 100                       # walls-only plan
                fp = straight_line_collision(st, plan_path(st, goal, furn, extent, levels=furn_levels),
                                             furn, extent) * 100                       # furniture-aware plan
                s_str.append(cs); s_wallp.append(wl); s_furn.append(fp); n_pairs += 1
        tot_str += s_str; tot_pln += s_furn
        print(f"{sc}: {len(s_str)} routes | furniture-collision: straight {np.mean(s_str):.1f}% | "
              f"walls-only plan {np.mean(s_wallp):.1f}% | furniture-aware plan {np.mean(s_furn):.2f}%")

    if n_pairs:
        ts, tp = np.array(tot_str), np.array(tot_pln)
        print(f"\n=== ALL {n_pairs} routes, FURNITURE-collision (walls + non-target furniture) ===")
        print(f"straight line          : mean {ts.mean():.1f}%  max {ts.max():.1f}%")
        print(f"furniture-aware planner: mean {tp.mean():.2f}%  max {tp.max():.2f}%")
        hit = ts > 1.0
        if hit.any():
            print(f"on the {hit.sum()} routes that hit furniture: straight {ts[hit].mean():.1f}% "
                  f"-> planned {tp[hit].mean():.2f}%  (the chair-hit, fixed)")


if __name__ == "__main__":
    main()
