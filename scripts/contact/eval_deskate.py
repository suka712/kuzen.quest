#!/usr/bin/env python3
"""Evaluate the foot-deskating cleanup (src/foot_contact.deskate) with oracle discipline (risk #4).

Checks, over N HUMANISE walk clips, three motion sources (GT / VQ-VAE recon / production gen):
  1. ORACLE: GT through deskate must NOT increase skate and must barely move (GT is already clean).
     If deskating GT inflates skate or drifts the body, the method is wrong -- stop.
  2. PAYOFF: recon & gen skate must DROP toward GT.
  3. NO DISTORTION: bone-length drift ~0 (length-preserving re-projection), root/pelvis path
     unchanged (goal error preserved exactly), no NEW penetration introduced.
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
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import yup_to_zup  # noqa: E402
from rollout import build_cond, load_model  # noqa: E402
from foot_contact import deskate, foot_metrics, bone_length_drift_mm, FPS  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")


def joints(d263):
    return yup_to_zup(mf.recover_positions(np.asarray(d263, np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/wander_data/step10/checkpoints/goalaug"))
    ap.add_argument("--vqvae", default=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gen", action="store_true")
    args = ap.parse_args()

    net = load_vqvae(args.vqvae, device=DEV)
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    trans = ns = cmodel = None
    if not args.no_gen:
        cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
        trans, ns = load_model(args.ckpt)

    flat = build_flat_join()
    idxs = [i for i, p in enumerate(flat) if p["action"] == "walk"]
    rng = np.random.RandomState(args.seed); rng.shuffle(idxs)

    # per source: lists of (before_metrics, after_metrics, bone_drift_mm, root_shift_mm)
    rec_out = {"GT": [], "recon": [], "gen": []}

    def record(src, J):
        Jd = deskate(J)
        rec_out[src].append((
            foot_metrics(J), foot_metrics(Jd), bone_length_drift_mm(J, Jd),
            float(np.linalg.norm(Jd[:, 0, :2] - J[:, 0, :2], axis=-1).max() * 1000),  # pelvis xy shift mm
        ))

    with torch.no_grad():
        for idx in idxs:
            if len(rec_out["GT"]) >= args.n:
                break
            rec = get_record(int(idx))
            try:
                cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
                d0, *_ = mf.humanise_positions_to_263(cm)
            except Exception:
                continue
            if d0.shape[0] < 20:
                continue
            d0 = d0.astype(np.float32)
            Jg = joints(d0)
            if np.linalg.norm(Jg[-1, 0, :2] - Jg[0, 0, :2]) < 0.8:
                continue
            record("GT", Jg)
            xin = torch.from_numpy((d0 - mean) / std).unsqueeze(0).to(DEV)
            d_rec = net.forward_decoder(net.encode(xin))[0].cpu().numpy() * std + mean
            record("recon", joints(d_rec))
            if not args.no_gen:
                fb = os.path.join(BEV, f"{rec.scene}.npz")
                if not os.path.exists(fb):
                    continue
                zb = np.load(fb); occ = zb["occ"].astype(np.float32); extent = zb["extent"]
                _, xy, _, sincos = compute_track2(rec)
                start_pose = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
                prefix = mf.local_joint_positions(d0)[0].ravel().astype(np.float32)
                act = "walk" if ns["cond_mode"] in ("full_action", "full_action_head") else None
                extra = build_cond(ns["cond_mode"], xy[-1].astype(np.float32), start_pose, prefix, occ,
                                   extent, np.array(ns["cond_mean"], np.float32),
                                   np.array(ns["cond_std"], np.float32), action=act)
                feat = cmodel.encode_text(clip.tokenize([rec.utterance], truncate=True).to(DEV)).float()
                cond = torch.cat([feat, torch.from_numpy(extra).unsqueeze(0).to(DEV)], -1)
                tokg = trans.sample(cond, if_categorial=False)
                if tokg.numel() == 0:
                    continue
                d_gen = net.forward_decoder(tokg)[0].cpu().numpy() * std + mean
                record("gen", joints(d_gen))
            if len(rec_out["GT"]) % 10 == 0:
                print(f"...{len(rec_out['GT'])} clips", flush=True)

    print(f"\n=== deskate over {len(rec_out['GT'])} walk clips (mm) ===")
    hdr = f"{'source':7s} {'skate→ mm/s':>18s} {'pen_max b→a':>14s} {'bone_drift':>11s} {'root_shift':>11s}"
    print(hdr)
    for src in ("GT", "recon", "gen"):
        rows = rec_out[src]
        if not rows:
            continue
        sb = np.mean([b["skate_mm"] for b, a, _, _ in rows]) * FPS
        sa = np.mean([a["skate_mm"] for b, a, _, _ in rows]) * FPS
        pb = np.mean([b["pen_max_mm"] for b, a, _, _ in rows])
        pa = np.mean([a["pen_max_mm"] for b, a, _, _ in rows])
        drift = np.max([d for _, _, d, _ in rows])
        rshift = np.max([r for _, _, _, r in rows])
        print(f"{src:7s} {sb:7.0f} → {sa:6.0f}     {pb:5.1f} → {pa:5.1f}   {drift:8.3f}   {rshift:8.3f}   (n={len(rows)})")
    print("\nPASS if: GT skate not increased & root_shift 0; recon/gen skate drops toward GT's;")
    print("         bone_drift ~0 (length-preserving); root_shift 0 (goal preserved).")


if __name__ == "__main__":
    main()
