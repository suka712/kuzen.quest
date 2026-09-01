#!/usr/bin/env python3
"""Stage A' -- finetune the geometry-grounded (heightmap-conditioned) VQ-VAE (SceMoS port).

Warm-starts from the scene-BLIND finetuned tokenizer (track2 net_iter020000.pth, RESULTS §3) and
wraps it in SceneVQVAE (src/scene_vqvae.py): encoder+quantizer untouched, a per-frame local
heightmap fused into the decoder input (identity-init so recon == the warm-start at iter 0). Adds
a penetration loss (src/contact_loss.py). Trained jointly on H3D (flat-floor heightmap) + HUMANISE
(real per-frame heightmap), balanced, same recons/commit/vel recipe as the scene-blind finetune.

Gate (Stage A'): per-category MPJPE must not regress AND the counterfactual follow-ratio (does the
decoded pelvis track a raised surface?) must be clearly > 0 -- the mechanism the scene-blind
tokenizer cannot have. Eval via eval_scene_tokenizer.run_scene_eval.

Two param groups: the pretrained base gets a low LR (don't wreck the converged codebook/decoder),
the NEW heightmap pathway (hm_encoder + fusion) a higher LR so it can learn to use the heightmap.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "track2"))

import joint_vqvae_dataset as jd  # noqa: E402
import scene_joint_dataset as sjd  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from scene_vqvae import SceneVQVAE  # noqa: E402
from contact_loss import penetration_loss  # noqa: E402
import eval_per_category_mpjpe as evalcat  # noqa: E402
import eval_scene_tokenizer as scene_eval  # noqa: E402
from train_vqvae_joint_finetune import (  # noqa: E402  reuse the tested finetune helpers
    prepare_quantizer_for_finetune, update_lr_warm_up, Heartbeat,
)

T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/dsp52026/Khiem/T2M-GPT")
if T2M_GPT_ROOT not in sys.path:
    sys.path.insert(0, T2M_GPT_ROOT)
import utils.losses as losses  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", default="scene_vqvae")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/wander_data/scene_tokenizer/checkpoints"))
    ap.add_argument("--base-vqvae", default=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"))
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--h3d-frac", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--lr-base", type=float, default=2e-5)
    ap.add_argument("--lr-new", type=float, default=2e-4)
    ap.add_argument("--warm-up-iter", type=int, default=200)
    ap.add_argument("--lr-scheduler", type=int, nargs="+", default=[15000])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--commit", type=float, default=0.02)
    ap.add_argument("--loss-vel", type=float, default=0.5)
    ap.add_argument("--contact-weight", type=float, default=0.5)
    ap.add_argument("--contact-margin", type=float, default=0.03)
    # Vertical-shift augmentation: make the heightmap a COMMAND, not a correlate (analog of
    # goal-aug §10). Encode the ORIGINAL clip (token unchanged), but shift the heightmap by Delta
    # and require the decoded body to shift by Delta -- the token cannot predict Delta, only the
    # heightmap carries it, so the decoder is forced to read the surface. Without this the decoder
    # ignores the (redundant-on-GT) heightmap: follow_ratio stays 0 (measured, RESULTS §11 trap).
    ap.add_argument("--height-aug", type=float, default=0.5, help="prob of shifting a sample")
    ap.add_argument("--haug-lo", type=float, default=-0.3)
    ap.add_argument("--haug-hi", type=float, default=0.5)
    ap.add_argument("--consist-weight", type=float, default=1.0,
                    help="shift-consistency loss weight: forces encode(motion)~=encode(motion+delta) "
                         "so tokens carry no absolute contact height (the heightmap does)")
    ap.add_argument("--recons-loss", default="l1_smooth")
    ap.add_argument("--c-h", type=int, default=128)
    ap.add_argument("--total-iter", type=int, default=20000)
    ap.add_argument("--eval-iter", type=int, default=2500)
    ap.add_argument("--print-iter", type=int, default=100)
    ap.add_argument("--heartbeat-sec", type=float, default=15.0)
    ap.add_argument("--eval-n-clips", type=int, default=120)
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    hb = Heartbeat(os.path.join(out_dir, "heartbeat.log"), args.heartbeat_sec)
    hb.write(f"START exp={args.exp_name} args={vars(args)}")

    mean = np.load(evalcat.EVAL_MEAN_PATH).astype(np.float32)
    std = np.load(evalcat.EVAL_STD_PATH).astype(np.float32)
    mean_t = torch.from_numpy(mean).to(DEVICE)
    std_t = torch.from_numpy(std).to(DEVICE)

    print("Loading H3D train split...")
    h3d_motions, h3d_stats = jd.load_h3d_split("train", window_size=args.window_size)
    print("Loading HUMANISE train split (263 + heightmaps)...")
    hum_motions, hum_hms, hum_stats = sjd.load_humanise_split_with_hm("train", window_size=args.window_size)
    hb.write(f"data loaded h3d={h3d_stats} humanise={hum_stats}")

    h3d_ds = sjd.HMWindowDataset(h3d_motions, None, mean, std, window_size=args.window_size)
    hum_ds = sjd.HMWindowDataset(hum_motions, hum_hms, mean, std, window_size=args.window_size)
    loader = sjd.BalancedJointHMLoader(h3d_ds, hum_ds, batch_size=args.batch_size,
                                       h3d_frac=args.h3d_frac, num_workers=args.num_workers,
                                       seed=args.seed)
    loader_iter = iter(loader)

    base = load_vqvae(ckpt_path=args.base_vqvae, device=DEVICE)
    prepare_quantizer_for_finetune(base)   # EMA-reset landmine (RESULTS §3)
    scene = SceneVQVAE(base, c_h=args.c_h).to(DEVICE)

    # UNFROZEN: encoder + quantizer(codebook) + decoder + hm pathway all train. The frozen-encoder
    # version made tokens carry absolute contact height, which dominated the heightmap at GENERATION
    # (RESULTS §12). Here the shift-consistency loss (step() below) forces the encoder to be
    # vertical-shift invariant, so tokens drop absolute height and the heightmap becomes the sole
    # contact-height source. Codebook adapts (EMA in train mode) -> tokens WILL change -> Stage 7
    # re-extracts + retrains the transformer.
    def _set_train():
        scene.train()
    _set_train()
    hb.write(f"model: SceneVQVAE around {args.base_vqvae}; UNFROZEN (encoder+quantizer+decoder+"
             f"hm pathway), shift-consistency; tokens WILL change (re-extract after)")

    new_params = list(scene.hm_encoder.parameters()) + list(scene.fusion.parameters())
    new_ids = {id(p) for p in new_params}
    base_params = [p for p in scene.parameters() if id(p) not in new_ids]   # enc+quant+dec
    optimizer = optim.AdamW([
        {"params": base_params, "lr": args.lr_base},
        {"params": new_params, "lr": args.lr_new},
    ], betas=(0.9, 0.99), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler,
                                                     gamma=args.gamma)
    Loss = losses.ReConsLoss(args.recons_loss, 22)
    base_lrs = [args.lr_base, args.lr_new]

    def save_ckpt(name, nb_iter):
        path = os.path.join(out_dir, name)
        torch.save({"net": scene.state_dict(), "iter": nb_iter}, path)
        return path

    def run_eval(nb_iter):
        scene.eval()
        t0 = time.time()
        res = scene_eval.run_scene_eval(scene, mean, std, n_clips=args.eval_n_clips,
                                        verbose=True, do_counterfactual=True)
        _set_train()
        with open(os.path.join(out_dir, f"eval_iter{nb_iter:06d}.json"), "w") as f:
            json.dump({"iter": nb_iter, "results": res, "sec": time.time() - t0}, f, indent=2)
        summ = " ".join(f"{k}={v.get('mean_mm', float('nan')):.0f}" for k, v in res.items())
        foll = " ".join(f"{k}.follow={v['follow_ratio']:.2f}" for k, v in res.items()
                        if "follow_ratio" in v)
        hb.write(f"EVAL iter={nb_iter} mpjpe[{summ}] {foll}")
        return res

    print("iter-0 eval (identity init == scene-blind base)...")
    run_eval(0)
    save_ckpt("net_iter000000.pth", 0)

    # 22 height channels of the 263 vector (root_y + each joint's y); shifting these by Delta is
    # an exact rigid vertical translation of all 22 joints (verified: horizontal unchanged).
    HEIGHT_IDX = torch.tensor([3] + [5 + 3 * k for k in range(21)], device=DEVICE)
    std_h = std_t[HEIGHT_IDX]   # for shifting in normalized space

    def step(gt_motion, gt_hm):
        B = gt_motion.shape[0]
        if args.height_aug > 0:
            do = (torch.rand(B, device=DEVICE) < args.height_aug).float()
            delta = (torch.rand(B, device=DEVICE) * (args.haug_hi - args.haug_lo)
                     + args.haug_lo) * do                      # (B,) metres, 0 where not augmented
        else:
            delta = torch.zeros(B, device=DEVICE)
        hm_aug = gt_hm + delta[:, None, None, None]            # surface shifts with Delta
        x_shift = gt_motion.clone()
        x_shift[:, :, HEIGHT_IDX] += (delta[:, None, None] / std_h)   # body shifts with Delta
        # Shift-consistency: encode BOTH the original and the shifted motion; tie their latents so
        # the encoder becomes vertical-shift invariant (tokens carry no absolute height). Decode the
        # SHIFTED motion from its own latent against the shifted heightmap, which now supplies the
        # absolute contact height. At token-extraction time we encode the unshifted motion -> a
        # height-agnostic token.
        z_orig = scene.encode_latent(gt_motion)
        pred, loss_commit, ppl, z_shift = scene.forward_from_motion(x_shift, hm_aug)
        loss_consist = F.mse_loss(z_orig, z_shift)
        loss_motion = Loss(pred, x_shift)
        loss_vel = Loss.forward_vel(pred, x_shift)
        loss_pen = penetration_loss(pred, hm_aug, mean_t, std_t, margin=args.contact_margin)
        loss = (loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
                + args.contact_weight * loss_pen + args.consist_weight * loss_consist)
        return loss, loss_motion, ppl, loss_commit, loss_pen, loss_consist

    print("Warmup...")
    t_start = time.time()
    for nb_iter in range(1, args.warm_up_iter + 1):
        for gi, lr in enumerate(base_lrs):
            cur = lr * nb_iter / (args.warm_up_iter + 1)
            optimizer.param_groups[gi]["lr"] = cur
        gm, gh = next(loader_iter)
        gm = gm.to(DEVICE).float(); gh = gh.to(DEVICE).float()
        loss, *_ = step(gm, gh)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        hb.maybe(nb_iter, extra=f"warmup loss={loss.item():.5f}")
    hb.write("WARMUP DONE")

    print(f"Main: {args.total_iter} iters")
    accR = accPen = accPPL = accCon = 0.0
    for nb_iter in range(1, args.total_iter + 1):
        gm, gh = next(loader_iter)
        gm = gm.to(DEVICE).float(); gh = gh.to(DEVICE).float()
        loss, lmo, ppl, lc, lpen, lcon = step(gm, gh)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        accR += lmo.item(); accPen += lpen.item(); accPPL += ppl.item(); accCon += lcon.item()
        hb.maybe(nb_iter, extra=f"train loss={loss.item():.4f} pen={lpen.item():.4f} "
                                f"consist={lcon.item():.4f} it/s={nb_iter/(time.time()-t_start):.2f}")
        if nb_iter % args.print_iter == 0:
            n = args.print_iter
            print(f"it {nb_iter}/{args.total_iter} recons={accR/n:.5f} pen={accPen/n:.5f} "
                  f"consist={accCon/n:.5f} ppl={accPPL/n:.1f} "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}/{optimizer.param_groups[1]['lr']:.1e} "
                  f"({nb_iter/(time.time()-t_start):.2f} it/s)")
            accR = accPen = accPPL = accCon = 0.0
        if nb_iter % args.eval_iter == 0 or nb_iter == args.total_iter:
            save_ckpt("net_last.pth", nb_iter)
            save_ckpt(f"net_iter{nb_iter:06d}.pth", nb_iter)
            run_eval(nb_iter)
    hb.write("TRAINING COMPLETE")
    print("done.")


if __name__ == "__main__":
    main()
