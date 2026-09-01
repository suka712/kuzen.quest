#!/usr/bin/env python3
"""Foot-contact quality of generated LOCOMOTION (placement-side contact-projection, step 1b).

Sit contact-HEIGHT headroom is small on HUMANISE (measure_contact_headroom.py replicated RESULTS
§12). The other half of "contact" is FOOT contact during walking: foot-skate (a planted foot
sliding), penetration (foot below the floor), float (stance foot hovering). VQ-VAE/T2M-GPT motion
is known to foot-skate from token discretization. Unlike sit-height, this applies to EVERY clip and
needs no furniture, so if the headroom is real a cleanup is universally justified.

Scene-free: everything is in the CLIP-FLOOR frame (process_file sets floor = min joint height over
the clip => feet rest at ~0), so foot height IS height above the floor with no mesh sampling.

Three sources isolate where skate comes from:
  GT     canonical HUMANISE motion (recover_positions of the stored 263) -- the data's own floor.
  recon  encode(GT 263) -> decode  -- pure VQ-VAE round trip, no transformer. Isolates the tokenizer.
  gen    the production step-11 model generating a walk to the GT endpoint. The real demo path.

ORACLE (risk #4): GT is the control. If GT itself shows large skate/penetration, the METRIC or the
floor convention is wrong, not the model -- read no model number until GT is clean.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for p in ["src", "scripts/track1", "scripts/scene_probe", "scripts/scene_tokenizer", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import clip  # noqa: E402
import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import yup_to_zup, world_to_local_xy  # noqa: E402
from rollout import build_cond, load_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
J_LFOOT, J_RFOOT = 10, 11
FPS = 20
H_CONTACT = 0.05   # foot within 5 cm of the floor = candidate contact
V_CONTACT = 0.02   # and moving < 2 cm/frame = planted (standard skate-metric gate)


def joints_from_263(d263):
    """canonical (T,263) -> (T,22,3) Z-up, feet ~0 (clip floor)."""
    return yup_to_zup(mf.recover_positions(np.asarray(d263, np.float32)))


def foot_metrics(J):
    """J (T,22,3) clip-floor Z-up. Returns dict of foot-skate / penetration / float (all mm)."""
    T = J.shape[0]
    skate_disp, pen_frame, float_frame = [], [], []
    for jf in (J_LFOOT, J_RFOOT):
        h = J[:, jf, 2]
        xy = J[:, jf, :2]
        spd = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1)])
        contact = (h < H_CONTACT) & (spd < V_CONTACT)
        for t in range(1, T):
            if contact[t] and contact[t - 1]:
                skate_disp.append(np.linalg.norm(xy[t] - xy[t - 1]))
    lf = np.minimum(J[:, J_LFOOT, 2], J[:, J_RFOOT, 2])  # lower foot each frame
    pen = np.maximum(0.0, -lf)                            # below floor
    # float = lower-foot height when NEITHER foot is near the floor (both hovering)
    both_up = lf > H_CONTACT
    return dict(
        skate_mm=float(np.mean(skate_disp) * 1000) if skate_disp else 0.0,
        skate_n=len(skate_disp),
        pen_mean_mm=float(pen.mean() * 1000),
        pen_max_mm=float(pen.max() * 1000),
        float_mm=float(lf[both_up].mean() * 1000) if both_up.any() else 0.0,
        float_frac=float(both_up.mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/wander_data/step10/checkpoints/goalaug"),
                    help="navigation model for the gen source (goalaug walks best)")
    ap.add_argument("--vqvae", default=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gen", action="store_true", help="skip the transformer gen source (faster)")
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

    agg = {"GT": [], "recon": [], "gen": []}
    with torch.no_grad():
        for idx in idxs:
            if len(agg["GT"]) >= args.n:
                break
            rec = get_record(int(idx))
            try:
                cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
                d0, *_ = mf.humanise_positions_to_263(cm)
            except Exception:
                continue
            if d0.shape[0] < 20:      # want a real walk, not a 2-step stub
                continue
            d0 = d0.astype(np.float32)
            J_gt = joints_from_263(d0)
            # displacement filter: only clips that actually travel (foot-skate needs walking)
            disp = np.linalg.norm(J_gt[-1, 0, :2] - J_gt[0, 0, :2])
            if disp < 0.8:
                continue
            agg["GT"].append(foot_metrics(J_gt))

            xin = torch.from_numpy((d0 - mean) / std).unsqueeze(0).to(DEV)
            tok = net.encode(xin)
            d_rec = net.forward_decoder(tok)[0].cpu().numpy() * std + mean
            agg["recon"].append(foot_metrics(joints_from_263(d_rec)))

            if not args.no_gen:
                fb = os.path.join(BEV, f"{rec.scene}.npz")
                if not os.path.exists(fb):
                    agg["gen"].append(None); continue
                zb = np.load(fb); occ = zb["occ"].astype(np.float32); extent = zb["extent"]
                _, xy, _, sincos = compute_track2(rec)
                start_pose = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
                goal = xy[-1].astype(np.float32)
                prefix = mf.local_joint_positions(d0)[0].ravel().astype(np.float32)
                extra = build_cond(ns["cond_mode"], goal, start_pose, prefix, occ, extent,
                                   np.array(ns["cond_mean"], np.float32), np.array(ns["cond_std"], np.float32),
                                   action="walk" if ns["cond_mode"] in ("full_action", "full_action_head") else None)
                feat = cmodel.encode_text(clip.tokenize([rec.utterance], truncate=True).to(DEV)).float()
                cond = torch.cat([feat, torch.from_numpy(extra).unsqueeze(0).to(DEV)], -1)
                tokg = trans.sample(cond, if_categorial=False)
                if tokg.numel() == 0:
                    agg["gen"].append(None); continue
                d_gen = net.forward_decoder(tokg)[0].cpu().numpy() * std + mean
                agg["gen"].append(foot_metrics(joints_from_263(d_gen)))
            if len(agg["GT"]) % 10 == 0:
                print(f"...{len(agg['GT'])} clips", flush=True)

    print(f"\n=== foot contact over {len(agg['GT'])} walk clips (clip-floor frame, mm) ===")
    print(f"{'source':8s} {'skate mm/fr':>12s} {'(mm/s)':>8s} {'pen_mean':>9s} {'pen_max':>8s} "
          f"{'float_mm':>9s} {'float%':>7s}")
    for src in ("GT", "recon", "gen"):
        rows = [r for r in agg[src] if r is not None]
        if not rows:
            continue
        def m(k):
            return float(np.mean([r[k] for r in rows]))
        print(f"{src:8s} {m('skate_mm'):12.1f} {m('skate_mm')*FPS:8.0f} {m('pen_mean_mm'):9.1f} "
              f"{m('pen_max_mm'):8.1f} {m('float_mm'):9.1f} {m('float_frac')*100:6.0f}%  (n={len(rows)})")
    print("\nread: recon >> GT skate => the TOKENIZER injects foot-skate (universal headroom);")
    print("      gen ~ recon => generation adds little beyond the round trip.")


if __name__ == "__main__":
    main()
