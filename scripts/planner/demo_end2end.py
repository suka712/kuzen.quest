#!/usr/bin/env python3
"""END-TO-END demo (build-order step 13): natural-language instruction + a ScanNet room -> a local
VLM plan -> steered, interacting motion -> full-mesh video. This is the whole pipeline in one script.

  instruction + scene image
    --(qwen3.5:27b, scene_anchors)--> ordered plan {action, target anchor}         [qwen_plan.py]
    --(expand_plan)--> per-segment (text, action, world goal), walk-ups auto-split
    --(guided rollout, step-11 action model + guided_seg steering on walks)--> world motion
    --(render_mesh_demo.render_in_mesh)--> mp4 in the real textured room mesh

The VLM does the grounding (which anchor); geometry gives the coordinate; the motion model is TOLD the
action + goal per segment (CLAUDE.md 2c). Walk segments are collision-steered (RESULTS §13); sit/lie/
stand segments are decoded greedily (steering is meaningless for an in-place interaction).
"""
import argparse
import os
import sys

import clip
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining", "scripts/planner"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import load_model, build_cond, yaw_from_joints, HEAD_MIN_DISP  # noqa: E402
from se2_utils import se2_place_full_body  # noqa: E402
from demo_rollout import sample_waypoints  # noqa: E402
from collision_guided import safe_sample, decode_place, path_collision  # noqa: E402
from render_mesh_demo import render_in_mesh, stitch  # noqa: E402
from qwen_plan import make_plan  # noqa: E402
from scene_anchors import load_scene_maps  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_HOP = 1.1        # per walk segment (RESULTS §9): longer goals undershoot
FRONT = 0.35         # stop this far before an interaction target so the sit goal is in-distribution


def expand_plan(plan, start_xy, occ, extent, rng):
    """Plan segments -> flat per-rollout (texts, actions, goals). A 'walk' to a target is split into
    <=MAX_HOP hops that DELIVER the body; a walk that precedes a sit/lie on the same target stops
    FRONT m short so the interaction goal is short (compose_goals_texts logic, plan-driven)."""
    texts, actions, goals, kinds = [], [], [], []
    cur = np.asarray(start_xy, float)
    approach_u = np.array([1.0, 0.0])
    n = len(plan)
    for i, seg in enumerate(plan):
        act = seg["action"]
        if act == "walk":
            if seg["target"] == "away":
                wp = sample_waypoints(occ, extent, cur, 1, min_step=1.0, rng=rng, max_step=1.9)
                dest = np.asarray(wp[0], float) if wp else cur + approach_u * 1.4
            else:
                dest = np.asarray(seg["xy"], float)
                d = cur - dest; nrm = np.linalg.norm(d)
                approach_u = d / (nrm if nrm > 1e-6 else 1e-6)
                if i + 1 < n and plan[i + 1]["action"] in ("sit", "lie") \
                        and plan[i + 1].get("target") == seg["target"]:
                    dest = dest + FRONT * approach_u          # stop just in front of the furniture
            span = np.linalg.norm(dest - cur)
            nh = max(1, int(np.ceil(span / MAX_HOP)))
            for k in range(nh):
                goals.append(cur + (dest - cur) * ((k + 1) / nh))
                texts.append("walk to the target"); actions.append("walk"); kinds.append("walk")
            cur = dest
        elif act in ("sit", "lie"):
            dest = np.asarray(seg["xy"], float)
            texts.append("sit on the couch" if act == "sit" else "lie on the bed")
            actions.append(act); goals.append(dest); kinds.append(act); cur = dest
        elif act == "stand up":
            dest = cur + 0.45 * approach_u                    # step back off the furniture
            texts.append("stand up from the couch")
            actions.append("stand up"); goals.append(dest); kinds.append("stand up"); cur = dest
    return texts, actions, goals, kinds


def guided_rollout(trans, net, cmodel, mean, std, ns, texts, actions, goals, start_pose, prefix,
                   occ, extent, tall, n_cand=8, coll_weight=10.0, steer=True):
    """Chain the expanded plan. Walk segments are collision-steered (guided_seg, RESULTS §13);
    interaction segments are decoded greedily. reorient applies to walks only."""
    cmean = np.array(ns["cond_mean"], np.float32); cstd = np.array(ns["cond_std"], np.float32)
    use_act = ns["cond_mode"] in ("full_action", "full_action_head")
    pose = np.asarray(start_pose, np.float32).copy(); pfx = np.asarray(prefix, np.float32).copy()
    segs = []
    with torch.no_grad():
        for txt, act, goal, in zip(texts, actions, goals):
            is_walk = act == "walk"
            if is_walk:                                       # reorient to face travel (walks only)
                dvec = np.asarray(goal, float) - pose[:2]
                if np.linalg.norm(dvec) >= HEAD_MIN_DISP:
                    ry = float(np.arctan2(dvec[1], dvec[0]))
                    pose = pose.copy(); pose[2], pose[3] = np.sin(ry), np.cos(ry)
            feat = cmodel.encode_text(clip.tokenize([txt], truncate=True).to(DEV)).float()
            extra = build_cond(ns["cond_mode"], np.asarray(goal, float), pose, pfx, occ, extent,
                               cmean, cstd, action=act if use_act else None)
            cond = torch.cat([feat, torch.from_numpy(extra).unsqueeze(0).to(DEV)], -1)
            nc = n_cand if (is_walk and steer) else 1
            best = None
            for j in range(nc):
                tok = safe_sample(trans, cond, if_categorial=(j > 0))
                if tok.numel() == 0:
                    continue
                world, local = decode_place(net, tok, pose, mean, std)
                ge = float(np.linalg.norm(world[-1, 0, :2] - np.asarray(goal, float)))
                cl = path_collision(world[:, 0, :2], tall, extent)
                score = ge + coll_weight * cl
                if best is None or score < best[0]:
                    best = (score, world, local, ge, cl)
            if best is None:
                break
            _, world, local, ge, cl = best
            segs.append({"world": world, "action": act, "goal_err": ge, "coll": cl})
            end_xy = world[-1, 0, :2]; end_yaw = yaw_from_joints(world[-1])
            pose = np.array([end_xy[0], end_xy[1], np.sin(end_yaw), np.cos(end_yaw)], np.float32)
            pfx = local[-1].ravel().astype(np.float32)
    return segs


def standing_prefix():
    """A neutral standing frame-0 pose to seed the chain (any walk clip's first frame)."""
    flat = build_flat_join()
    for i, p in enumerate(flat):
        if p["action"] != "walk":
            continue
        try:
            cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{i:05d}.npy"))
            d0, *_ = mf.humanise_positions_to_263(cm)
            if d0.shape[0] >= 8:
                return mf.local_joint_positions(d0.astype(np.float32))[0].ravel().astype(np.float32)
        except Exception:
            continue
    raise RuntimeError("no standing prefix found")


def free_start(occ, extent, anchors, rng, min_from_furniture=1.2):
    """Sample a free-floor start away from the anchors, so segment 1 is a real walk-up."""
    for _ in range(200):
        wp = sample_waypoints(occ, extent, np.array([extent[0], extent[2]]), 1,
                              min_step=0.5, rng=rng, max_step=99.0)
        if not wp:
            continue
        p = np.asarray(wp[0], float)
        if all(np.linalg.norm(p - a["xy"]) > min_from_furniture for a in anchors):
            return p
    return np.array([(extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default=None, help="start x,y world meters (default: auto free floor)")
    ap.add_argument("--no-steer", dest="steer", action="store_false", default=True)
    ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--coll-weight", type=float, default=10.0)
    ap.add_argument("--blend-n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.RandomState(args.seed)

    rgb, occ, tall, extent = load_scene_maps(args.scene)
    occ = occ.astype(np.float32); tall = tall.astype(np.float32)

    # --- PLAN (VLM) ---
    from scene_anchors import detect_anchors
    anchors = detect_anchors(occ.astype(bool), tall.astype(bool), extent)
    start_xy = np.array([float(v) for v in args.start.split(",")]) if args.start \
        else free_start(occ, extent, anchors, rng)
    img_path = os.path.join(args.out, f"plan_{args.scene}.png")
    plan, anchors, raw = make_plan(args.scene, args.instruction, start_xy, img_path)
    print(f"\nINSTRUCTION: {args.instruction}\nSTART: ({start_xy[0]:.2f},{start_xy[1]:.2f})")
    print("VLM PLAN:")
    for i, s in enumerate(plan):
        loc = "away/open floor" if s["target"] == "away" else \
              f"#{s['target']} ({s['xy'][0]:.2f},{s['xy'][1]:.2f})"
        print(f"  {i+1}. {s['action']:9s} -> {loc}")
    if not plan:
        print("empty plan; aborting"); return

    # --- MOTION ---
    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)

    texts, actions, goals, kinds = expand_plan(plan, start_xy, occ, extent, rng)
    d = np.asarray(goals[0]) - start_xy; yaw = float(np.arctan2(d[1], d[0]))
    start_pose = np.array([start_xy[0], start_xy[1], np.sin(yaw), np.cos(yaw)], np.float32)
    prefix = standing_prefix()
    print(f"\nexpanded into {len(texts)} rollout segments "
          f"({sum(k=='walk' for k in kinds)} walk, {sum(k not in ('walk',) for k in kinds)} interaction)")

    segs = guided_rollout(trans, net, cmodel, mean, std, ns, texts, actions, goals, start_pose,
                          prefix, occ, extent, tall, n_cand=args.n_cand,
                          coll_weight=args.coll_weight, steer=args.steer)
    if len(segs) < len(texts):
        print(f"WARNING: only {len(segs)}/{len(texts)} segments generated")
    if not segs:
        print("no motion generated"); return
    path = np.concatenate([s["world"][:, 0, :2] for s in segs])
    print(f"path {float(np.sum(np.linalg.norm(np.diff(path,axis=0),axis=1))):.1f} m  "
          f"collision {path_collision(path, tall, extent)*100:.1f}%")
    # risk #4: goal error is z-blind, so verify interactions by PELVIS HEIGHT, not goal error
    from humanise_join import J_PELVIS
    for i, (k, s) in enumerate(zip(kinds, segs)):
        if k in ("sit", "lie", "stand up"):
            z = s["world"][:, J_PELVIS, 2]
            verdict = ("SAT" if z[-1] < 0.7 else "did NOT sit") if k in ("sit", "lie") \
                else ("STOOD" if z[-1] > 0.8 else "did NOT stand")
            print(f"  seg {i} [{k}]: pelvis end {z[-1]:.2f} m (min {z.min():.2f}) -> {verdict}")

    # --- RENDER ---
    allw = stitch(segs, args.blend_n)
    out_path = os.path.join(args.out, f"end2end_{args.scene}_{args.seed}.mp4")
    n = render_in_mesh(args.scene, allw, out_path, fps=20, follow=True)
    print(f"rendered {n} frames -> {out_path}")


if __name__ == "__main__":
    main()
