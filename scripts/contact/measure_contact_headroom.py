#!/usr/bin/env python3
"""Contact-height HEADROOM measurement (placement-side contact-projection, step 1 / risk #4).

Question: on the REAL production pipeline (step-11 action transformer + the BASE scene-blind
VQ-VAE, NO scene tokenizer), how far is a generated seated pelvis from the ACTUAL seat surface,
and does that error DEPEND ON SEAT HEIGHT? RESULTS §12 found the aggregate contact-height gap
small BUT noted the scene-blind decoder sits at a ~constant nominal height "regardless of the real
seat", which implies a LARGE error on non-nominal (tall / low) furniture that the aggregate hides.
This script measures that split before any corrector is built.

Everything is in the CLIP-FLOOR frame (height above the local ground), the SAME frame se2_place
and scene_heightmap.to_clip_frame already use, so seat height and body Z are directly comparable
(eval_contact_demo.py conventions, reused).

ORACLE (mandatory, risk #4): the GT clip's own seated pelvis, placed the identical way. A correctly
seated pelvis rests ~0.156 m above the seat (RESULTS §12, hip joint above the cushion). If the GT
oracle does not land near seat+0.156, the seat-height measurement itself is wrong and no model
number off it is trustworthy.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/scene_probe", "scripts/scene_tokenizer", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import clip  # noqa: E402
import motion_features as mf  # noqa: E402
from humanise_join import (build_flat_join, get_record, compute_track2,  # noqa: E402
                           J_PELVIS)
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import se2_place_full_body  # noqa: E402
from scene_heightmap import local_heightmap, to_clip_frame, GRID_N  # noqa: E402
from scene_decode import build_scene_context, local_ground  # noqa: E402
from rollout import build_cond, load_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
J_LFOOT, J_RFOOT = 10, 11        # SMPL 22-joint toe joints (lowest points)
FIRED_Z = 0.75                    # a sit "fired" if the seated pelvis is below this (else it walked/stood)
GT_HIP_ABOVE_SEAT = 0.156         # RESULTS §12


def seat_height_at(scene_ctx, sit_xy, facing, floor_ref):
    """Seat surface directly under the body, clip-floor-referenced (identical to eval_contact_demo)."""
    kdt, verts_z, _ = scene_ctx
    h_abs, _ = local_heightmap(kdt, verts_z, np.asarray(sit_xy, float), float(facing))
    hm = to_clip_frame(h_abs, floor_ref)
    c = GRID_N // 2
    return float(np.median(hm[c - 1:c + 2, c - 1:c + 2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/wander_data/step11/checkpoints/action"))
    ap.add_argument("--vqvae", default=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--front", type=float, default=0.6, help="standing spot distance in front of seat")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = load_vqvae(args.vqvae, device=DEV)
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)
    cmean = np.array(ns["cond_mean"], np.float32); cstd = np.array(ns["cond_std"], np.float32)

    flat = build_flat_join()
    idxs = [i for i, p in enumerate(flat) if p["action"] == "sit"]
    rng = np.random.RandomState(args.seed); rng.shuffle(idxs)

    rows = []
    with torch.no_grad():
        for idx in idxs:
            if len(rows) >= args.n:
                break
            rec = get_record(int(idx))
            fb = os.path.join(BEV, f"{rec.scene}.npz")
            if not os.path.exists(fb):
                continue
            zb = np.load(fb); occ = zb["occ"].astype(np.float32); extent = zb["extent"]
            try:
                cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
                d0, *_ = mf.humanise_positions_to_263(cm)
            except Exception:
                continue
            if d0.shape[0] < 8:
                continue
            _, xy, _, sincos = compute_track2(rec)
            sit_xy = xy[-1].astype(np.float32)
            seat_face = float(np.arctan2(sincos[-1, 0], sincos[-1, 1]))
            try:
                scene_ctx = build_scene_context(rec.scene)
            except Exception:
                continue
            # standing spot in front of the seat, facing it (matches eval_contact_demo)
            d = sit_xy - xy[0, :2]
            n = np.linalg.norm(d); u = d / (n if n > 1e-6 else 1e-6)
            s_xy = sit_xy - args.front * u
            yaw = float(np.arctan2(u[1], u[0]))
            start_pose = np.array([s_xy[0], s_xy[1], np.sin(yaw), np.cos(yaw)], np.float32)
            floor_ref = local_ground(scene_ctx, start_pose[:2])
            seat_h = seat_height_at(scene_ctx, sit_xy, seat_face, floor_ref)

            # --- GT ORACLE: place the GT clip at its own start pose; seated pelvis should be seat+0.156
            gt_start = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
            w_gt = se2_place_full_body(d0.astype(np.float32), gt_start, mf)
            pelvis_gt = float(w_gt[-1, J_PELVIS, 2])

            # --- MODEL: production step-11 action model + base VQ-VAE (scene-blind)
            prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel().astype(np.float32)
            extra = build_cond(ns["cond_mode"], sit_xy, start_pose, prefix, occ, extent,
                               cmean, cstd, action="sit")
            feat = cmodel.encode_text(clip.tokenize([rec.utterance], truncate=True).to(DEV)).float()
            cond = torch.cat([feat, torch.from_numpy(extra).unsqueeze(0).to(DEV)], -1)
            tok = trans.sample(cond, if_categorial=False)
            if tok.numel() == 0:
                continue
            motion = net.forward_decoder(tok)[0].cpu().numpy() * std + mean
            w = se2_place_full_body(motion.astype(np.float32), start_pose, mf)
            pelvis_m = float(w[-1, J_PELVIS, 2])
            foot_m = float(min(w[-1, J_LFOOT, 2], w[-1, J_RFOOT, 2]))  # lowest foot at the seated frame

            rows.append(dict(scene=rec.scene, seat_h=seat_h, pelvis_gt=pelvis_gt,
                             pelvis_m=pelvis_m, foot_m=foot_m,
                             fired=pelvis_m < FIRED_Z))
            print(f"[{len(rows):2d}] {rec.scene} seat={seat_h:+.2f}  GT_pelvis={pelvis_gt:.2f}"
                  f" (GT-seat={pelvis_gt-seat_h:+.2f})  model_pelvis={pelvis_m:.2f}"
                  f" (m-seat={pelvis_m-seat_h:+.2f})  fired={rows[-1]['fired']}", flush=True)

    if not rows:
        print("no rows"); return
    arr = rows
    fired = [r for r in arr if r["fired"]]
    print(f"\n=== headroom: {len(arr)} sits, {len(fired)} fired. correct seated pelvis = seat+{GT_HIP_ABOVE_SEAT:.3f} ===")

    def summarize(subset, label):
        if len(subset) < 2:
            print(f"[{label:24s}] n={len(subset)} (too few)"); return
        seat = np.array([r["seat_h"] for r in subset])
        pgt = np.array([r["pelvis_gt"] for r in subset])
        pm = np.array([r["pelvis_m"] for r in subset])
        # contact error = |pelvis - (seat + 0.156)|; signed = pelvis - seat (positive = above seat)
        err_gt = np.abs(pgt - seat - GT_HIP_ABOVE_SEAT)
        err_m = np.abs(pm - seat - GT_HIP_ABOVE_SEAT)
        print(f"[{label:24s}] n={len(subset):2d} seat[{seat.min():+.2f},{seat.max():+.2f}]  "
              f"GT contact-err {err_gt.mean():.3f} (oracle~0)  |  MODEL contact-err {err_m.mean():.3f}  "
              f"model pelvis {pm.mean():.2f}±{pm.std():.2f}")

    summarize(fired, "ALL fired")
    summarize([r for r in fired if r["seat_h"] < 0.40], "LOW seats (<0.40)")
    summarize([r for r in fired if 0.40 <= r["seat_h"] <= 0.60], "NOMINAL seats (0.40-0.60)")
    summarize([r for r in fired if r["seat_h"] > 0.60], "TALL seats (>0.60)")

    # is the model's seated pelvis ~constant regardless of seat (the §12 claim)?
    seat = np.array([r["seat_h"] for r in fired]); pm = np.array([r["pelvis_m"] for r in fired])
    if len(fired) > 2 and seat.std() > 1e-6:
        print(f"\ncorr(seat, model_pelvis) = {np.corrcoef(seat, pm)[0,1]:+.2f}  "
              f"(near 0 => model ignores seat height => headroom is real)")
        print(f"corr(seat, GT_pelvis)    = {np.corrcoef(seat, np.array([r['pelvis_gt'] for r in fired]))[0,1]:+.2f}"
              f"  (near +1 => GT tracks the seat, oracle sane)")


if __name__ == "__main__":
    main()
