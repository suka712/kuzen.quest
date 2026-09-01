#!/usr/bin/env python3
"""Held-out eval for the geometry-grounded (heightmap-conditioned) VQ-VAE. Three questions:

  1. RECONSTRUCTION fidelity -- per-category MPJPE, heightmap-aware, must not regress vs the
     scene-blind finetuned tokenizer (RESULTS §3).
  2. PENETRATION of the reconstruction vs the GT heightmap (should stay ~0; the heightmap must
     not push the body through surfaces).
  3. COUNTERFACTUAL contact-following -- raise the local heightmap by +delta and decode the SAME
     tokens: does the decoded pelvis rise to track the new surface? This is the mechanism the
     scene-blind tokenizer structurally cannot have (no heightmap input), and the reason to grow
     the tokenizer at all. Reported as metres of pelvis rise per +0.2 m surface raise (1.0 =
     perfect following, 0.0 = ignores the surface).

Heightmaps: HUMANISE test clips load their extracted (clip-floor-referenced) heightmap; H3D uses
a flat floor (zeros). Frame alignment is offset 0 (verified 2026-08-28).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "track2"))
import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from scene_vqvae import SceneVQVAE  # noqa: E402
from scene_heightmap import GRID_N, flat_floor_heightmap  # noqa: E402
from contact_loss import penetration_loss  # noqa: E402
import eval_per_category_mpjpe as base_eval  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HM_CACHE = os.environ.get("WANDER_HUMANISE_HM_CACHE",
                          os.path.expanduser("~/wander_data/motion_data/HUMANISE_heightmap_cache"))
C263 = os.environ.get("WANDER_HUMANISE_263_CACHE",
                      os.path.expanduser("~/wander_data/motion_data/HUMANISE_263_cache"))


def _crop(T):
    return base_eval.crop_to_multiple(T)


def roundtrip_hm(scene, data263, hm, mean, std, mean_t, std_t):
    """Returns (mpjpe_m, pelvis_z_recon (T,), penetration_m) for one clip. hm aligned to data263."""
    T = _crop(data263.shape[0])
    if T < 4:
        return None
    d = data263[:T].astype(np.float32)
    h = hm[:T].astype(np.float32)
    x = torch.from_numpy((d - mean) / std).float().unsqueeze(0).to(DEVICE)
    ht = torch.from_numpy(h).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_out, _, _ = scene(x, ht)
        pen = penetration_loss(x_out, ht, mean_t, std_t).item()
    recon263 = x_out[0].cpu().numpy() * std + mean
    orig_local = mf.local_joint_positions(d)
    recon_local = mf.local_joint_positions(recon263.astype(np.float32))
    T2 = min(orig_local.shape[0], recon_local.shape[0])
    mpjpe = np.linalg.norm(orig_local[:T2] - recon_local[:T2], axis=-1).mean()
    pelvis_z = recon_local[:, 0, 1]  # root_y
    return float(mpjpe), pelvis_z, float(pen)


def counterfactual_follow(scene, data263, hm, mean, std, delta=0.2):
    """Raise the heightmap by +delta and re-decode the same tokens; return the mean rise of the
    decoded pelvis over the LATER half of the clip (where interaction contact is). Metres of rise
    per +delta surface raise; divide by delta for a 0..1 following ratio."""
    T = _crop(data263.shape[0])
    if T < 8:
        return None
    d = data263[:T].astype(np.float32)
    h = hm[:T].astype(np.float32)
    x = torch.from_numpy((d - mean) / std).float().unsqueeze(0).to(DEVICE)
    h0 = torch.from_numpy(h).float().unsqueeze(0).to(DEVICE)
    h1 = h0 + delta
    with torch.no_grad():
        o0, _, _ = scene(x, h0)
        o1, _, _ = scene(x, h1)
    r0 = (o0[0].cpu().numpy() * std + mean)
    r1 = (o1[0].cpu().numpy() * std + mean)
    z0 = mf.local_joint_positions(r0.astype(np.float32))[:, 0, 1]
    z1 = mf.local_joint_positions(r1.astype(np.float32))[:, 0, 1]
    half = T // 2
    return float(np.mean(z1[half:] - z0[half:]))


def _load_hm_for(idx, n_frames):
    p = os.path.join(HM_CACHE, f"{idx:05d}.npy")
    if not os.path.exists(p):
        return None
    h = np.load(p).astype(np.float32)
    if h.shape[0] != n_frames:
        return None
    return h


def run_scene_eval(scene, mean, std, n_clips=150, seed=0, delta=0.2, verbose=True,
                   do_counterfactual=True):
    mean_t = torch.from_numpy(mean).float().to(DEVICE)
    std_t = torch.from_numpy(std).float().to(DEVICE)
    results = {}

    # H3D reconstruction (flat-floor heightmap) -- fidelity must hold on locomotion
    rng = np.random.RandomState(seed)
    ids = base_eval.h3d_test_ids()
    sample = rng.choice(ids, size=min(n_clips, len(ids)), replace=False)
    errs = []
    for name in sample:
        d = np.load(f"{base_eval.H3D_ROOT}/new_joint_vecs/{name}.npy").astype(np.float32)
        T = _crop(d.shape[0])
        if T < 4:
            continue
        hm = np.zeros((T, GRID_N, GRID_N), dtype=np.float32)
        r = roundtrip_hm(scene, d, hm, mean, std, mean_t, std_t)
        if r:
            errs.append(r[0])
    results["h3d_baseline"] = {"mean_mm": float(np.mean(errs) * 1000), "n": len(errs)}

    # HUMANISE per-category (real heightmaps)
    by_action = base_eval.humanise_category_test_indices()
    for action in ["walk", "stand up", "sit", "lie"]:
        idxs = by_action.get(action, [])
        rng = np.random.RandomState(1)
        if idxs:
            chosen = rng.choice(idxs, size=min(n_clips, len(idxs)), replace=False)
        else:
            chosen = []
        errs, pens, follows = [], [], []
        for i in chosen:
            cm = np.load(f"{base_eval.HUMANISE_MOTIONS}/{int(i):05d}.npy")
            if cm.shape[0] < 6:
                continue
            d263, _, _, _ = mf.humanise_positions_to_263(cm)
            d263 = d263.astype(np.float32)
            hm = _load_hm_for(int(i), d263.shape[0])
            if hm is None:
                continue
            r = roundtrip_hm(scene, d263, hm, mean, std, mean_t, std_t)
            if r is None:
                continue
            errs.append(r[0])
            pens.append(r[2])
            if do_counterfactual and action in ("sit", "lie"):
                f = counterfactual_follow(scene, d263, hm, mean, std, delta)
                if f is not None:
                    follows.append(f)
        key = f"humanise_{action.replace(' ', '_')}"
        results[key] = {
            "mean_mm": float(np.mean(errs) * 1000) if errs else float("nan"),
            "penetration_mm": float(np.mean(pens) * 1000) if pens else float("nan"),
            "n": len(errs),
        }
        if follows:
            results[key]["follow_ratio"] = float(np.mean(follows) / delta)
    if verbose:
        _print(results)
    return results


def _print(results):
    print("=" * 84)
    print(f"{'category':<20} {'mpjpe(mm)':>10} {'pen(mm)':>9} {'follow':>8} {'n':>5}")
    for k, v in results.items():
        print(f"{k:<20} {v.get('mean_mm', float('nan')):>10.1f} "
              f"{v.get('penetration_mm', float('nan')):>9.1f} "
              f"{v.get('follow_ratio', float('nan')):>8.2f} {v.get('n', 0):>5}")
    print("=" * 84)
    print("follow = pelvis rise per surface raise (1.0 tracks the surface, 0.0 ignores it)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SceneVQVAE checkpoint (.pth with 'net')")
    ap.add_argument("--base-vqvae", default=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"))
    ap.add_argument("--n-clips", type=int, default=150)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    mean = np.load(base_eval.EVAL_MEAN_PATH).astype(np.float32)
    std = np.load(base_eval.EVAL_STD_PATH).astype(np.float32)
    base = load_vqvae(args.base_vqvae, device=DEVICE)
    scene = SceneVQVAE(base).to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    scene.load_state_dict(ckpt["net"])
    scene.eval()
    print(f"Loaded SceneVQVAE from {args.ckpt}")
    results = run_scene_eval(scene, mean, std, n_clips=args.n_clips, delta=args.delta)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"ckpt": args.ckpt, "results": results}, f, indent=2)
        print(f"saved -> {args.out_json}")


if __name__ == "__main__":
    main()
