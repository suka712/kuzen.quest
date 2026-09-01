#!/usr/bin/env python3
"""Orientation-driven "true sit" demo (Stage 8): the APPROACH + sit facing are planned from the
PERCEIVED furniture facing, NOT from ground truth. This is the legitimate planner-side fix for the
sit-orientation limitation (RESULTS §11, IN_FLIGHT): the motion model can't learn facing (HUMANISE
bakes in approach≈sit-facing), so orientation must come from perception feeding the planner.

Mechanic:
  1. Perceive the seat's forward direction F ("away from the backrest tall-mass",
     probe_furniture_orientation.perceive_facing). If ambiguous (no backrest) -> SKIP (curation).
  2. Place the start on the FRONT (F) side of the seat; the walk-up approaches the seat front.
  3. Reorient the SIT segment to face F (seg_headings) -- the "turn around and sit facing out". The
     seam stays clean because the prefix pose is heading-canonicalized (RESULTS §11 reorient).
No GT is used to drive anything; GT sit-facing is used ONLY to SCORE the result.

Reports per clip: |sit facing - F| (did the sit obey the planned facing?) and |sit facing - GT|
(is it actually right?), plus SAT/STOOD (pelvis height). --render writes mp4s.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import clip
import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2, J_PELVIS  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import rollout, load_model, yaw_from_joints  # noqa: E402
from demo_interaction import compose_goals_texts, object_phrase, pelvis_z, collision, render  # noqa: E402
from probe_furniture_orientation import perceive_facing  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
TALL = os.path.expanduser("~/wander_data/bev_tall_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def wrapdeg(a):
    return abs((np.degrees(a) + 180) % 360 - 180)


def _free(occ, extent, xy):
    """True if the world point is walkable floor (not occupied in the 0.12 m occupancy raster)."""
    xmin, xmax, ymin, ymax = extent
    H, W = occ.shape
    c = int(np.clip((xy[0] - xmin) / (xmax - xmin) * W, 0, W - 1))
    r = int(np.clip((ymax - xy[1]) / (ymax - ymin) * H, 0, H - 1))
    return occ[r, c] < 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-demos", type=int, default=10)
    ap.add_argument("--start-dist", type=float, default=2.2)
    ap.add_argument("--front", type=float, default=0.35)
    ap.add_argument("--radius", type=float, default=0.7, help="perceive_facing backrest radius")
    ap.add_argument("--sit-z", type=float, default=0.7)
    ap.add_argument("--stand-z", type=float, default=0.8)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--blend-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)

    flat = build_flat_join()
    seed_idx = [i for i, p in enumerate(flat) if p["action"] == "sit"]
    rng = np.random.RandomState(args.seed); rng.shuffle(seed_idx)

    made = ambiguous = 0
    rows = []
    for idx in seed_idx:
        if made >= args.n_demos:
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
        sit_xy = xy[-1].astype(np.float32)
        gt_face = float(np.arctan2(sincos[-1, 0], sincos[-1, 1]))

        # PERCEIVE the seat facing (no GT). Skip ambiguous furniture (curation).
        F = perceive_facing(tall, extent, sit_xy, r_m=args.radius)
        if F is None:
            ambiguous += 1
            continue
        Fvec = np.array([np.cos(F), np.sin(F)], np.float32)

        # Approach the seat IN direction F (from the -F/back side) so the sit fires forward-and-down
        # AND naturally faces F -- no turn needed (the model sits facing its approach). This only
        # works where the back side is walkable (freestanding furniture); if -F is occupied, skip.
        s_xy = sit_xy - args.start_dist * Fvec
        if not _free(occ, extent, s_xy) or not _free(occ, extent, sit_xy - 0.6 * Fvec):
            continue  # back side blocked (wall-backed seat) -> can't approach facing F
        approach_yaw = F  # face forward (+F), toward and onto the seat
        start_pose = np.array([s_xy[0], s_xy[1], np.sin(approach_yaw), np.cos(approach_yaw)], np.float32)
        prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel().astype(np.float32)

        goals, nwi = compose_goals_texts(start_pose[:2], sit_xy, occ, extent, rng,
                                         front=args.front, stand=0.5, away=1.3)
        obj = object_phrase(rec.utterance)
        texts = [f"walk to {obj}"] * nwi + [rec.utterance, f"stand up from {obj}", "walk to the door"]
        actions = ["walk"] * nwi + ["sit", "stand up", "walk"]
        seg_headings = None  # the sit inherits the +F approach facing -- no forced turn

        segs = rollout(trans, net, cmodel, clip, mean, std, ns, texts, goals, start_pose, prefix,
                       occ, extent,
                       actions=actions if ns["cond_mode"] in ("full_action", "full_action_head") else None,
                       reorient=True, seg_headings=seg_headings)
        if len(segs) < len(goals):
            continue

        zs = [pelvis_z(s) for s in segs]
        sat = float(zs[nwi][-1]) < args.sit_z
        stood = float(zs[nwi + 1][-1]) > args.stand_z
        sit_seg = segs[nwi]
        sit_face = yaw_from_joints(sit_seg["world"][-1])
        err_F = wrapdeg(sit_face - F)          # did the sit obey the planned facing?
        err_GT = wrapdeg(sit_face - gt_face)   # is it actually correct?
        perc_vs_gt = wrapdeg(F - gt_face)      # how good was the perception itself?
        rows.append(dict(idx=idx, scene=rec.scene, obj=obj, sat=sat, stood=stood,
                         err_F=err_F, err_GT=err_GT, perc_vs_gt=perc_vs_gt, segs=segs,
                         goals=goals, texts=texts, sit_xy=sit_xy, occ=occ, extent=extent, tall=tall))
        made += 1
        print(f"[{made}] {rec.scene} {obj:22s} SAT={sat} STOOD={stood} | perceived-vs-GT={perc_vs_gt:3.0f}deg "
              f"sit-obeys-plan={err_F:3.0f}deg sit-vs-GT={err_GT:3.0f}deg", flush=True)

    if not rows:
        print(f"no demos ({ambiguous} skipped as ambiguous)"); return
    egt = np.array([r["err_GT"] for r in rows]); ef = np.array([r["err_F"] for r in rows])
    pvg = np.array([r["perc_vs_gt"] for r in rows])
    n_sat = sum(r["sat"] and r["stood"] for r in rows)
    print(f"\n=== {len(rows)} orientation-driven sits ({ambiguous} skipped ambiguous), "
          f"{n_sat} SAT&STOOD ===")
    print(f"perception (perceived F vs GT):   median {np.median(pvg):.0f}deg  <45deg {100*(pvg<45).mean():.0f}%")
    print(f"sit obeys the planned facing:     median {np.median(ef):.0f}deg  <30deg {100*(ef<30).mean():.0f}%")
    print(f"sit facing vs GT (end to end):    median {np.median(egt):.0f}deg  <45deg {100*(egt<45).mean():.0f}%")
    print("=> if 'obeys plan' is small, the seg_heading fix works; end-to-end is capped by perception.")

    if args.render:
        for r in rows:
            if not (r["sat"] and r["stood"]):
                continue
            p = os.path.join(args.out, f"sit_{r['idx']}_{r['scene']}_gt{r['err_GT']:.0f}.mp4")
            try:
                render(r["segs"], r["texts"], r["occ"], r["extent"], r["tall"], r["goals"],
                       r["sit_xy"], p, f"{r['scene']} {r['obj']} (sit-vs-GT {r['err_GT']:.0f}deg)",
                       args.blend_n)
                print(f"  rendered {p}", flush=True)
            except Exception as e:
                print(f"  render failed {r['idx']}: {e}")


if __name__ == "__main__":
    main()
