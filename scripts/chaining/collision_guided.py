#!/usr/bin/env python3
"""Collision-guided decoding (build-order step 12) — the last SWAPPABLE contribution.

The problem (RESULTS §8, IN_FLIGHT): the model follows goals but does NOT steer. On chained
free-waypoint rollouts it collides with the 0.9 m tall-obstacle map ~2.15% of the path — MORE
than a straight line between the same waypoints (~1.57%). Nothing in the system avoids anything;
it just walks between waypoints, clipping furniture the polyline also clips.

This script compares, on the SAME scenes / waypoints / seed (a fair head-to-head, CLAUDE.md
risk #4 — the straight-line polyline is the oracle control that makes a collision % readable):

  greedy       the current baseline: argmax decoding (rollout.py, if_categorial=False).
  reject_chain the guaranteed FLOOR (CLAUDE.md 2e): generate N whole stochastic chains,
               keep the one with the lowest total path collision. Always improves
               non-collision; costs Nx generation and picks globally, not reactively.
  guided_seg   the actual guided decoder (CLAUDE.md 2e "segment-level rejection sampling"):
               at EACH segment, sample N candidate token sequences, decode+SE(2)-place each,
               and keep the one minimizing  goal_err + w * collision. Locally reactive (it
               re-plans every ~1 m hop around what is actually in front of it) and cheap
               (N per segment, not N**6). goal_err stays IN the objective so it cannot "avoid"
               a wall by refusing to move — the trap CLAUDE.md warns about.

WHY STOCHASTIC IS REQUIRED. rollout()'s sample(if_categorial=False) is greedy argmax — fully
deterministic, so N draws are identical and rejection sampling is a no-op. Both guided modes
draw from Categorical(probs) (if_categorial=True) to get PATH diversity to select over. The
greedy argmax is always included as candidate 0 in guided_seg, so guided_seg is never worse
than greedy on its own objective.
"""
import argparse
import os
import sys

import clip
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import load_model, build_cond, yaw_from_joints, HEAD_MIN_DISP  # noqa: E402
from se2_utils import se2_place_full_body  # noqa: E402
from demo_rollout import sample_waypoints  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
TALL = os.path.expanduser("~/wander_data/bev_tall_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def path_collision(path_xy, tall, extent):
    """Fraction of the (N,2) world path that lands on the 0.9 m tall-obstacle raster."""
    xmin, xmax, ymin, ymax = extent
    H, W = tall.shape
    c = np.clip(((path_xy[:, 0] - xmin) / (xmax - xmin) * W).astype(int), 0, W - 1)
    r = np.clip(((ymax - path_xy[:, 1]) / (ymax - ymin) * H).astype(int), 0, H - 1)
    return float((tall[r, c] > 0.5).mean())


def straight_line_collision(start_xy, wps, tall, extent, step=0.05):
    """Oracle control: collision of the waypoint POLYLINE walked directly (no model)."""
    poly = [np.asarray(start_xy, float)]
    for w in wps:
        a0, b0 = poly[-1], np.asarray(w, float)
        k = max(2, int(np.linalg.norm(b0 - a0) / step))
        poly += [a0 + (b0 - a0) * t for t in np.linspace(0, 1, k)[1:]]
    return path_collision(np.stack(poly), tall, extent)


def safe_sample(trans, cond, if_categorial):
    """trans.sample() has an upstream bug (t2m_trans.py:60): a stochastic draw that emits the
    end-token at position 0 leaves `xs` unbound and raises UnboundLocalError. That candidate is
    just a degenerate empty segment -- treat it as no tokens so the caller skips it."""
    try:
        tok = trans.sample(cond, if_categorial=if_categorial)
    except UnboundLocalError:
        return torch.zeros((1, 0), dtype=torch.long, device=DEV)
    return tok


def decode_place(net, tok, pose, mean, std):
    """tok -> (world (T,22,3) Z-up, local joints (T,22,3)) placed onto `pose`."""
    motion = (net.forward_decoder(tok)[0].cpu().numpy() * std + mean).astype(np.float32)
    world = se2_place_full_body(motion, pose, mf)
    local = mf.local_joint_positions(motion)
    return world, local


def _reorient_pose(pose, goal):
    """Rotate the start heading to face the goal (walk-scale moves only) — the inference
    heading fix (RESULTS §11). Identical to rollout(reorient=True)."""
    dvec = np.asarray(goal, float) - pose[:2]
    if np.linalg.norm(dvec) >= HEAD_MIN_DISP:
        ry = float(np.arctan2(dvec[1], dvec[0]))
        pose = pose.copy(); pose[2], pose[3] = np.sin(ry), np.cos(ry)
    return pose


def _handoff(world, local):
    """Next segment's start pose (decoded ending) and prefix (decoded ending body config)."""
    end_xy = world[-1, 0, :2]
    end_yaw = yaw_from_joints(world[-1])
    pose = np.array([end_xy[0], end_xy[1], np.sin(end_yaw), np.cos(end_yaw)], np.float32)
    prefix = local[-1].ravel().astype(np.float32)
    return pose, prefix


def run_chain(trans, net, cmodel, mean, std, ns, texts, goals, start_pose, prefix,
              occ, extent, tall, reorient, mode, n_cand, coll_weight, rng):
    """One chained rollout under `mode`. Returns list of per-segment dicts (world, goal_err,
    coll of the chosen segment). guided_seg selects per segment; greedy/stochastic just decode.

    mode: 'greedy'      -> argmax, 1 candidate/seg (the baseline).
          'stochastic'  -> Categorical draw, 1 candidate/seg (a single reject_chain sample).
          'guided_seg'  -> n_cand candidates/seg, keep argmin(goal_err + coll_weight*coll)."""
    cmean = np.array(ns["cond_mean"], np.float32)
    cstd = np.array(ns["cond_std"], np.float32)
    pose = np.asarray(start_pose, np.float32).copy()
    pfx = np.asarray(prefix, np.float32).copy()
    segs = []
    with torch.no_grad():
        for txt, goal in zip(texts, goals):
            if reorient:
                pose = _reorient_pose(pose, goal)
            feat = cmodel.encode_text(clip.tokenize([txt], truncate=True).to(DEV)).float()
            extra = build_cond(ns["cond_mode"], np.asarray(goal, float), pose, pfx,
                               occ, extent, cmean, cstd)
            cond = torch.cat([feat, torch.from_numpy(extra).unsqueeze(0).to(DEV)], -1)

            if mode == "guided_seg":
                # candidate 0 = greedy argmax (guarantees guided is never worse than greedy on
                # its own score); the rest are stochastic draws that give PATH diversity to
                # steer over. Decode+place each, score goal_err + w*collision, keep the best.
                cands = []
                for j in range(max(1, n_cand)):
                    tok = safe_sample(trans, cond, if_categorial=(j > 0))
                    if tok.numel() == 0:
                        continue
                    world, local = decode_place(net, tok, pose, mean, std)
                    ge = float(np.linalg.norm(world[-1, 0, :2] - np.asarray(goal, float)))
                    cl = path_collision(world[:, 0, :2], tall, extent)
                    cands.append((ge + coll_weight * cl, ge, cl, world, local))
                if not cands:
                    break
                _, ge, cl, world, local = min(cands, key=lambda t: t[0])
            else:
                tok = safe_sample(trans, cond, if_categorial=(mode == "stochastic"))
                if tok.numel() == 0:
                    break
                world, local = decode_place(net, tok, pose, mean, std)
                ge = float(np.linalg.norm(world[-1, 0, :2] - np.asarray(goal, float)))
                cl = path_collision(world[:, 0, :2], tall, extent)

            segs.append({"world": world, "goal_err": ge, "seg_coll": cl})
            pose, pfx = _handoff(world, local)
    return segs


def chain_metrics(segs, tall, extent):
    path = np.concatenate([s["world"][:, 0, :2] for s in segs])
    return {"coll": path_collision(path, tall, extent),
            "goal": float(np.mean([s["goal_err"] for s in segs])),
            "dist": float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rollouts", type=int, default=20)
    ap.add_argument("--n-segments", type=int, default=6)
    ap.add_argument("--min-step", type=float, default=0.6)
    ap.add_argument("--max-step", type=float, default=1.2)
    ap.add_argument("--n-cand", type=int, default=8, help="candidates per segment (guided_seg) "
                    "and whole chains (reject_chain)")
    ap.add_argument("--coll-weight", type=float, default=10.0,
                    help="weight on collision in the guided_seg per-segment score "
                         "goal_err(m) + w*coll_frac. Swept {2,5,10,20} over 2 seeds: collision "
                         "falls monotonically with w while goal error stays flat (~0.10 m, best-of-N); "
                         "10 beats the straight-line oracle on both seeds without a goal-error cost.")
    ap.add_argument("--reorient", action="store_true", default=True,
                    help="face-travel heading fix (on by default; matches the demo)")
    ap.add_argument("--no-reorient", dest="reorient", action="store_false")
    ap.add_argument("--modes", default="greedy,reject_chain,guided_seg",
                    help="comma list subset of greedy,reject_chain,guided_seg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    modes = args.modes.split(",")
    torch.manual_seed(args.seed)  # stochastic draws come from torch's RNG (Categorical)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)
    print(f"model cond_mode={ns['cond_mode']}  modes={modes}  n_cand={args.n_cand}  "
          f"coll_weight={args.coll_weight}  reorient={args.reorient}\n", flush=True)

    flat = build_flat_join()
    walk_idx = [i for i, p in enumerate(flat) if p["action"] == "walk"]
    rng = np.random.RandomState(args.seed); rng.shuffle(walk_idx)

    agg = {m: {"coll": [], "goal": []} for m in modes}
    agg["line"] = {"coll": []}
    done = 0
    for idx in walk_idx:
        if done >= args.n_rollouts:
            break
        rec = get_record(int(idx))
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
        # one waypoint set per scene, shared by every mode + the oracle (fair comparison)
        wps = sample_waypoints(occ, extent, start_pose[:2], args.n_segments, args.min_step,
                               rng, max_step=args.max_step)
        if wps is None:
            continue
        texts = ["walk to the target"] * args.n_segments

        line_coll = straight_line_collision(start_pose[:2], wps, tall, extent)
        row = {"line": line_coll}
        ok = True
        for m in modes:
            if m == "reject_chain":
                best = None
                for _ in range(args.n_cand):
                    segs = run_chain(trans, net, cmodel, mean, std, ns, texts, wps, start_pose,
                                     prefix, occ, extent, tall, args.reorient, "stochastic",
                                     1, 0.0, rng)
                    if len(segs) < args.n_segments:
                        continue
                    mt = chain_metrics(segs, tall, extent)
                    if best is None or mt["coll"] < best["coll"]:
                        best = mt
                if best is None:
                    ok = False; break
                mt = best
            else:
                segs = run_chain(trans, net, cmodel, mean, std, ns, texts, wps, start_pose,
                                 prefix, occ, extent, tall, args.reorient, m,
                                 args.n_cand, args.coll_weight, rng)
                if len(segs) < args.n_segments:
                    ok = False; break
                mt = chain_metrics(segs, tall, extent)
            row[m] = mt
        if not ok:
            continue
        for m in modes:
            agg[m]["coll"].append(row[m]["coll"]); agg[m]["goal"].append(row[m]["goal"])
        agg["line"]["coll"].append(line_coll)
        done += 1
        msg = "  ".join(f"{m}={row[m]['coll']*100:.1f}%" for m in modes)
        print(f"  [{done}/{args.n_rollouts}] {rec.scene}  line={line_coll*100:.1f}%  {msg}", flush=True)

    if done == 0:
        print("no rollouts produced"); return
    print(f"\n=== {done} rollouts of {args.n_segments} segments (seed {args.seed}) ===")
    print(f"  {'oracle straight-line':22s} collision {np.mean(agg['line']['coll'])*100:5.2f}%")
    for m in modes:
        print(f"  {m:22s} collision {np.mean(agg[m]['coll'])*100:5.2f}%   "
              f"goal {np.mean(agg[m]['goal']):.3f} m")
    np.save(os.path.join(args.out, f"cg_seed{args.seed}.npy"),
            np.array([agg], dtype=object), allow_pickle=True)
    print(f"\nsaved {os.path.join(args.out, f'cg_seed{args.seed}.npy')}")


if __name__ == "__main__":
    main()
