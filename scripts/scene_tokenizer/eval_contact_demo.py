#!/usr/bin/env python3
"""Stage 5 payoff metric: does the geometry-grounded decoder make SITS land on the ACTUAL seat
height, where the scene-blind decoder sits at a fixed nominal height regardless of the furniture?

Because the encoder+quantizer are frozen, the transformer generates the SAME tokens either way, so
this decodes one token sequence TWO ways -- scene-blind (net.forward_decoder) and heightmap
(scene_decode.decode_with_heightmap) -- and compares the seated pelvis height to the real seat
height sampled from the scene mesh. A contact-correct decoder's seated pelvis should TRACK the seat
height across scenes (correlation high, |pelvis - seat| low); the scene-blind one should be ~flat.

Single sit segment from a standing spot in front of real furniture (seeded from a real sit clip
for verified furniture + a known seat location), so there is no compounding from a walk-up.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/scene_probe", "scripts/chaining", "scripts/scene_tokenizer"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import clip  # noqa: E402
import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2, J_PELVIS  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from scene_vqvae import SceneVQVAE  # noqa: E402
from se2_utils import se2_place_full_body  # noqa: E402
from scene_heightmap import local_heightmap, to_clip_frame, GRID_N  # noqa: E402
from scene_decode import build_scene_context, decode_with_heightmap, local_ground  # noqa: E402
from rollout import build_cond, load_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
FIRED_Z = 0.75  # a sit "fired" if the scene-blind seated pelvis is below this (else it walked/stood)


def seat_height_at(scene_ctx, sit_xy, facing, floor_ref):
    kdt, verts_z, _ = scene_ctx
    h_abs, _ = local_heightmap(kdt, verts_z, np.asarray(sit_xy, float), float(facing))
    hm = to_clip_frame(h_abs, floor_ref)   # SAME reference the decoder uses (local ground)
    c = GRID_N // 2
    return float(np.median(hm[c - 1:c + 2, c - 1:c + 2]))  # seat surface directly under the body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="step-11 action transformer dir")
    ap.add_argument("--base-vqvae", default=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"))
    ap.add_argument("--scene-vqvae", required=True, help="trained SceneVQVAE .pth")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--front", type=float, default=0.6, help="standing spot distance in front of seat")
    ap.add_argument("--decode-iters", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = load_vqvae(args.base_vqvae, device=DEV)
    scene = SceneVQVAE(base).to(DEV)
    scene.load_state_dict(torch.load(args.scene_vqvae, map_location=DEV)["net"])
    scene.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)
    cmean = np.array(ns["cond_mean"], np.float32); cstd = np.array(ns["cond_std"], np.float32)
    if ns["cond_mode"] not in ("full_action", "full_action_head"):
        print("WARNING: transformer has no action conditioning; sit may not fire");

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
            cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
            try:
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
            # standing spot in front of the seat, facing it; standing prefix from the clip
            d = sit_xy - xy[0, :2]
            n = np.linalg.norm(d); u = d / (n if n > 1e-6 else 1e-6)
            s_xy = sit_xy - args.front * u
            yaw = float(np.arctan2(u[1], u[0]))
            start_pose = np.array([s_xy[0], s_xy[1], np.sin(yaw), np.cos(yaw)], np.float32)
            floor_ref = local_ground(scene_ctx, start_pose[:2])   # ground under the standing spot
            seat_h = seat_height_at(scene_ctx, sit_xy, seat_face, floor_ref)
            prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel().astype(np.float32)

            extra = build_cond(ns["cond_mode"], sit_xy, start_pose, prefix, occ, extent,
                               cmean, cstd, action="sit")
            feat = cmodel.encode_text(clip.tokenize([rec.utterance], truncate=True).to(DEV)).float()
            cond = torch.cat([feat, torch.from_numpy(extra).unsqueeze(0).to(DEV)], -1)
            tok = trans.sample(cond, if_categorial=False)
            if tok.numel() == 0:
                continue
            # decode the SAME tokens two ways, BOTH with the new scene decoder (the new tokens are a
            # new codebook -- the old base decoder would be garbage). "no-scene" = flat-floor
            # heightmap (what the height-agnostic token gives with no scene geometry); "heightmap" =
            # the real scene heightmap. If the port works, no-scene sits ~at the floor and heightmap
            # tracks the actual seat.
            Tg = tok.shape[-1] * 4
            flat = torch.zeros(1, Tg, GRID_N, GRID_N, device=DEV)
            m_sb = scene.forward_decoder(tok, flat)[0].cpu().numpy() * std + mean
            m_hm = decode_with_heightmap(scene, tok, start_pose, scene_ctx, mf, std, mean,
                                         n_iters=args.decode_iters)
            w_sb = se2_place_full_body(m_sb.astype(np.float32), start_pose, mf)
            w_hm = se2_place_full_body(m_hm.astype(np.float32), start_pose, mf)
            rows.append(dict(scene=rec.scene, seat_h=seat_h,
                             pelvis_sb=float(w_sb[-1, J_PELVIS, 2]),
                             pelvis_hm=float(w_hm[-1, J_PELVIS, 2])))
            print(f"[{len(rows)}] {rec.scene} seat={seat_h:.2f}  pelvis_sceneblind={rows[-1]['pelvis_sb']:.2f}"
                  f"  pelvis_heightmap={rows[-1]['pelvis_hm']:.2f}", flush=True)

    if not rows:
        print("no rows"); return

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-6 and b.std() > 1e-6 else float("nan")

    def report(subset, label):
        if len(subset) < 2:
            print(f"\n[{label}] n={len(subset)} (too few)"); return
        seat = np.array([r["seat_h"] for r in subset])
        psb = np.array([r["pelvis_sb"] for r in subset])
        phm = np.array([r["pelvis_hm"] for r in subset])
        print(f"\n[{label}] n={len(subset)}  seat range [{seat.min():.2f},{seat.max():.2f}]")
        print(f"  scene-blind: corr(seat) {corr(seat, psb):+.2f}  |pelvis-seat| {np.mean(np.abs(psb-seat)):.3f}")
        print(f"  heightmap  : corr(seat) {corr(seat, phm):+.2f}  |pelvis-seat| {np.mean(np.abs(phm-seat)):.3f}")

    fired = [r for r in rows if r["pelvis_sb"] < FIRED_Z]   # the sit actually happened
    print(f"\n=== does the seated pelvis track the ACTUAL seat height? ({len(rows)} sits, "
          f"{len(fired)} fired) ===")
    print("A seated pelvis rests ~on the seat surface, so |pelvis-seat| small AND corr(seat) high")
    print("means the sit lands on the real furniture. Scene-blind sits at a fixed nominal height.")
    report(fired, "FIRED sits (fair comparison)")
    report([r for r in fired if r["seat_h"] <= 0.60], "  fired, NOMINAL seats (<=0.60m)")
    report([r for r in fired if r["seat_h"] > 0.60], "  fired, TALL seats (>0.60m)")


if __name__ == "__main__":
    main()
