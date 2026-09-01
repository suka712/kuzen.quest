#!/usr/bin/env python3
"""Stage 1 of the geometry-grounded tokenizer: precompute the per-frame local heightmap for
every HUMANISE clip, aligned frame-for-frame with the 263 cache (src/scene_heightmap.py for the
definition; frame alignment verified 2026-08-28, offset 0).

Output: {CACHE_HM}/{idx:05d}.npy, shape (T263, 32, 32) float16, height-above-floor in [0, 2.0].
T263 is read from the 263 cache so the array is guaranteed to line up with what the VQ-VAE
trains on; compute_track2 supplies the world (x, y, yaw) per frame (its length is T263 + 1, we
take the first T263). Resumable (skips existing outputs). Per-scene KDTree cached across clips.

Why precompute (not on-the-fly in the dataloader): a 64-frame window needs 64x1024 KDTree
queries; doing that per __getitem__ across a multi-thousand-iteration finetune would dominate
runtime. ~1 hr one-off here, then O(load) at train time. H3D clips get a flat-floor heightmap
at load time (no scene) -- not precomputed here.
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO, "src"))
from humanise_join import build_flat_join, get_record, compute_track2, J_PELVIS  # noqa: E402
import bev_render  # noqa: E402
from scene_heightmap import build_scene_kdtree, local_heightmap, to_clip_frame, GRID_N  # noqa: E402

HUM = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE")
CACHE_263 = os.environ.get("WANDER_HUMANISE_263_CACHE",
                           os.path.expanduser("~/wander_data/motion_data/HUMANISE_263_cache"))
CACHE_HM = os.environ.get("WANDER_HUMANISE_HM_CACHE",
                          os.path.expanduser("~/wander_data/motion_data/HUMANISE_heightmap_cache"))
N_TOTAL = 19648


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=N_TOTAL)
    ap.add_argument("--out", default=CACHE_HM)
    ap.add_argument("--report-every", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene_cache = {}   # scene -> (kdt, verts_h, floor_z) or None

    def get_scene(scene):
        if scene not in scene_cache:
            if len(scene_cache) > 30:
                scene_cache.clear()
            try:
                mesh = bev_render._load_scene_mesh(scene)
                scene_cache[scene] = build_scene_kdtree(mesh.vertices)
            except Exception as e:  # noqa: BLE001
                print(f"  scene {scene} failed to load: {e}", flush=True)
                scene_cache[scene] = None
        return scene_cache[scene]

    t0 = time.time()
    n_done = n_skip_exist = n_skip_no263 = n_fail = 0
    fills = []
    flat = build_flat_join()  # warms the cache; also validates N_TOTAL
    for i in range(args.start, args.end):
        out_path = os.path.join(args.out, f"{i:05d}.npy")
        if os.path.exists(out_path):
            n_skip_exist += 1
            continue
        c263 = os.path.join(CACHE_263, f"{i:05d}.npy")
        if not os.path.exists(c263):
            n_skip_no263 += 1
            continue
        try:
            t263 = int(np.load(c263, mmap_mode="r").shape[0])
            rec = get_record(i)
            sc = get_scene(rec.scene)
            if sc is None:
                n_fail += 1
                continue
            kdt, verts_z, floor_z = sc
            jw, xy, yaw, _ = compute_track2(rec)   # (Ttrk,...), Ttrk == t263 + 1
            if xy.shape[0] < t263:
                n_fail += 1
                print(f"  #{i}: track {xy.shape[0]} < t263 {t263}, skipping", flush=True)
                continue
            # clip floor = min joint world Z over the USED frames (matches process_file's floor);
            # the stored heightmap is referenced to it so it shares the 263's vertical frame.
            clip_floor = float(jw[:t263, :, 2].min())
            hm = np.empty((t263, GRID_N, GRID_N), dtype=np.float16)
            f_acc = 0.0
            for t in range(t263):
                h_abs, f = local_heightmap(kdt, verts_z, xy[t], float(yaw[t]))
                hm[t] = to_clip_frame(h_abs, clip_floor).astype(np.float16)
                f_acc += f
            np.save(out_path, hm)
            fills.append(f_acc / t263)
            n_done += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  #{i} failed: {e}", flush=True)
            continue

        if (n_done + n_skip_exist) % args.report_every == 0:
            el = time.time() - t0
            rate = n_done / el if el > 0 else 0
            print(f"[{el:7.1f}s] i={i+1}/{args.end}  done={n_done} skip_exist={n_skip_exist} "
                  f"no263={n_skip_no263} fail={n_fail}  {rate:.1f} clips/s  "
                  f"fill~{np.mean(fills[-200:]) if fills else float('nan'):.2f}", flush=True)

    el = time.time() - t0
    print("=" * 60)
    print(f"DONE in {el:.1f}s  new={n_done} skip_exist={n_skip_exist} no263={n_skip_no263} "
          f"fail={n_fail}  mean_fill={np.mean(fills) if fills else float('nan'):.3f}")


if __name__ == "__main__":
    main()
