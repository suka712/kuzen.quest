#!/usr/bin/env python3
"""Stage 7a: re-extract tokens with the new (unfrozen, height-agnostic) tokenizer, reusing the
existing step-11 manifest's GEOMETRIC fields.

Only `tokens` depends on the tokenizer; prefix_pose, occ_crop, xy_traj, start, goal, action, text
are geometric (world track / scene / GT pose) and tokenizer-INDEPENDENT. So we load the existing
manifest, re-encode each clip's 263 with the new encoder+quantizer, and SWAP the tokens field,
keeping everything the transformer's full_action conditioning needs. Same normalization + crop as
prepare_probe_data (evaluator-consistent mean/std, crop_to_multiple), so the new token sequence
lines up 1:1 with the record it replaces.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "track1"))
import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from scene_vqvae import SceneVQVAE  # noqa: E402
from prepare_probe_data import crop_to_multiple  # noqa: E402

HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-vqvae", required=True)
    ap.add_argument("--base-vqvae", default=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"))
    ap.add_argument("--src-tokens", default=os.path.expanduser("~/wander_data/step10/tokens"),
                    help="existing manifest dir (train.pkl/test.pkl) to copy geometric fields from")
    ap.add_argument("--out", required=True, help="output tokens dir")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = load_vqvae(args.base_vqvae, device=DEV)
    scene = SceneVQVAE(base).to(DEV)
    scene.load_state_dict(torch.load(args.scene_vqvae, map_location=DEV)["net"])
    scene.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)

    n_changed = n_len = 0
    with torch.no_grad():
        for split in ["train", "test"]:
            with open(os.path.join(args.src_tokens, f"{split}.pkl"), "rb") as f:
                manifest = pickle.load(f)
            for rec in manifest:
                idx = rec["index"]
                cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
                d263, *_ = mf.humanise_positions_to_263(cm)
                old_tok = rec["tokens"]
                # CRITICAL: step10 tokenized the FIRST T frames where T = len(tokens)*4 (verified:
                # len(xy_traj)==len(tokens)*4 and xy_traj==xy[:T]). Reproduce that exact crop so the
                # new tokens describe the SAME motion the reused geometric fields (goal/prefix/
                # xy_traj) were computed for -- NOT crop_to_multiple(T263), which gave 2x too many.
                T = len(old_tok) * 4
                norm = (d263[:T].astype(np.float32) - mean) / std
                x = torch.from_numpy(norm).unsqueeze(0).to(DEV)
                new_tok = scene.encode(x)[0].cpu().numpy().astype(np.int64)
                if len(new_tok) != len(old_tok):
                    n_len += 1
                    L = min(len(new_tok), len(old_tok))
                    new_tok = new_tok[:L]
                if not np.array_equal(new_tok, old_tok[:len(new_tok)]):
                    n_changed += 1
                rec["tokens"] = new_tok
            with open(os.path.join(args.out, f"{split}.pkl"), "wb") as f:
                pickle.dump(manifest, f)
            print(f"{split}: {len(manifest)} clips re-tokenized -> {args.out}/{split}.pkl", flush=True)
    print(f"changed token seqs: {n_changed}  length-mismatch clips: {n_len} "
          f"(expect changed high -- new codebook; mismatch ~0)")


if __name__ == "__main__":
    main()
