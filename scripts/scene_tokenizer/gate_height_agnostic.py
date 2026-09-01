#!/usr/bin/env python3
"""GATE for the unfrozen shift-consistency tokenizer (Stage 6), BEFORE the expensive re-extract +
transformer retrain (Stage 7). Transformer-free.

The generation failure of the frozen tokenizer was: tokens carried absolute contact height, which
dominated the heightmap when the transformer emitted a generic sit token. The fix trains the
encoder to be vertical-shift invariant so tokens are HEIGHT-AGNOSTIC. Two checks decide whether it
worked:

 1. TOKEN SHIFT-INVARIANCE: encode(motion) vs encode(motion + Delta) -- fraction of identical
    tokens. High (=> the token dropped absolute height). This is the property the transformer
    retrain depends on: once tokens carry no absolute height, a generic generated sit token is
    height-agnostic and the heightmap is the only contact-height source.
 2. FOLLOW: decode the (now height-agnostic) token against hm and hm+delta; the seated pelvis must
    rise with the surface. With height-agnostic tokens this IS the generation behavior (unlike the
    frozen model, where follow was high but generation still failed because each token had a
    different baked-in height).

PASS if invariance is high AND follow ~1. Then re-extract tokens + retrain the transformer.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "track2"))
import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from scene_vqvae import SceneVQVAE  # noqa: E402
import eval_per_category_mpjpe as base_eval  # noqa: E402
from humanise_join import build_flat_join  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HM_CACHE = os.environ.get("WANDER_HUMANISE_HM_CACHE",
                          os.path.expanduser("~/wander_data/motion_data/HUMANISE_heightmap_cache"))
C263 = os.environ.get("WANDER_HUMANISE_263_CACHE",
                      os.path.expanduser("~/wander_data/motion_data/HUMANISE_263_cache"))
HEIGHT_IDX = np.array([3] + [5 + 3 * k for k in range(21)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-vqvae", required=True)
    ap.add_argument("--base-vqvae", default=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--follow-delta", type=float, default=0.2)
    args = ap.parse_args()

    mean = np.load(base_eval.EVAL_MEAN_PATH).astype(np.float32)
    std = np.load(base_eval.EVAL_STD_PATH).astype(np.float32)
    base = load_vqvae(args.base_vqvae, device=DEV)
    scene = SceneVQVAE(base).to(DEV)
    scene.load_state_dict(torch.load(args.scene_vqvae, map_location=DEV)["net"])
    scene.eval()

    flat = build_flat_join()
    test_ids = set(int(x) for x in base_eval._read_ids(f"{base_eval.HUMANISE_ROOT}/test.txt"))
    idxs = [i for i, p in enumerate(flat) if p["action"] in ("sit", "lie") and i in test_ids]
    rng = np.random.RandomState(0); rng.shuffle(idxs)

    agrees, follows = [], []
    with torch.no_grad():
        for i in idxs:
            if len(agrees) >= args.n:
                break
            p263 = f"{C263}/{i:05d}.npy"; phm = f"{HM_CACHE}/{i:05d}.npy"
            if not (os.path.exists(p263) and os.path.exists(phm)):
                continue
            m = np.load(p263).astype(np.float32); hm = np.load(phm).astype(np.float32)
            T = base_eval.crop_to_multiple(m.shape[0])
            if T < 8 or m.shape[0] != hm.shape[0]:
                continue
            m = m[:T]; hm = hm[:T]
            x = torch.from_numpy((m - mean) / std).float().unsqueeze(0).to(DEV)
            ms = m.copy(); ms[:, HEIGHT_IDX] += args.delta
            xs = torch.from_numpy((ms - mean) / std).float().unsqueeze(0).to(DEV)
            t0 = scene.encode(x); t1 = scene.encode(xs)
            agrees.append(float((t0 == t1).float().mean()))
            # follow: decode t0 with hm and hm+follow_delta
            ht = torch.from_numpy(hm).float().unsqueeze(0).to(DEV)
            r0 = scene.forward_decoder(t0, ht)[0].cpu().numpy() * std + mean
            r1 = scene.forward_decoder(t0, ht + args.follow_delta)[0].cpu().numpy() * std + mean
            z0 = mf.local_joint_positions(r0.astype(np.float32))[:, 0, 1]
            z1 = mf.local_joint_positions(r1.astype(np.float32))[:, 0, 1]
            follows.append(float(np.mean(z1[T // 2:] - z0[T // 2:]) / args.follow_delta))

    agrees = np.array(agrees); follows = np.array(follows)
    inv = agrees.mean(); fol = follows.mean()
    print(f"\n=== height-agnostic gate ({len(agrees)} sit/lie clips, shift Delta={args.delta}) ===")
    print(f"token shift-invariance: {inv:.2f}  (fraction of tokens identical for motion vs motion+Delta)")
    print(f"follow ratio          : {fol:.2f}  (seated pelvis rise per surface raise)")
    ok = inv > 0.80 and fol > 0.70
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} -- "
          + ("tokens are height-agnostic and the decoder follows the surface; proceed to "
             "re-extract + retrain the transformer." if ok else
             "tokens still carry absolute height (invariance low) or the decoder ignores the "
             "surface (follow low). Raise --consist-weight / retrain before the cascade."))


if __name__ == "__main__":
    main()
