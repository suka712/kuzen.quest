#!/usr/bin/env python3
"""Multi-segment rollout — chaining (build-order step 9).

Generates a chain of segments, each conditioned on the previous segment's
DECODED ending pose, placing each into the world with SE(2). This is the first
code in the project that produces motion longer than one segment.

THE THREE RULES IT OBEYS, all established by earlier stages:
 1. Goal is fed start-relative and heading-aligned (RESULTS §4). Absolute world
    coordinates do not work.
 2. The prefix is the ending BODY CONFIGURATION -- root-relative,
    heading-canonicalized joint positions, which are frame-independent -- not
    the previous segment's tokens, which describe a different canonical frame
    (RESULTS §7).
 3. The pose fed forward is the DECODED one, never a blended or smoothed one
    (RESULTS §7). An off-manifold prefix collapses the model: iid noise at 25mm
    costs 331mm of seam, while a VQ-VAE-reconstructed prefix at ~70mm costs 15.
    So when a seam blend is applied for display, it is applied to the OUTPUT
    only and never fed back as conditioning.

Next-segment heading is recovered from the generated body itself, using the
same hip/shoulder face-direction formula as humanise_join.compute_track2, so
the chain's own geometry decides where it is pointing rather than an assumption.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/scene_probe", "scripts/scene_tokenizer"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import J_LHIP, J_RHIP, J_LSHOULDER, J_RSHOULDER  # noqa: E402
from se2_utils import se2_place_full_body, world_to_local_xy  # noqa: E402
from train_probe import build_transformer, COND_EXTRA_DIMS, ACTION_IDS  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OCC_N = 28


def yaw_from_joints(p22):
    """World yaw from a single (22,3) Z-up pose, matching compute_track2's
    convention (yaw 0 = facing +X) so chained poses stay in one frame."""
    across = (p22[J_RHIP, :2] - p22[J_LHIP, :2]) + (p22[J_RSHOULDER, :2] - p22[J_LSHOULDER, :2])
    n = np.linalg.norm(across)
    across = across / (n if n > 1e-8 else 1e-8)
    fwd = np.array([-across[1], across[0]])
    return float(np.arctan2(fwd[1], fwd[0]))


def occ_crop(occ, extent, xy, yaw):
    from scene_probe import crop_agent_frame
    c = crop_agent_frame(occ, extent, xy, yaw)
    k = c.shape[0] // OCC_N
    return c.reshape(OCC_N, k, OCC_N, k).mean((1, 3)).ravel().astype(np.float32)


HEAD_MIN_DISP = 0.4  # below this displacement, don't command a turn (e.g. sitting in place)


def build_cond(cond_mode, goal_world, start_pose, prefix_pose, occ, extent, cmean, cstd,
               action=None, head_target=None):
    ACT = ("full_action", "full_action_head")
    parts = [world_to_local_xy(goal_world, start_pose).ravel()]
    if cond_mode in ("rel_prefix", "full") + ACT:
        parts.append(prefix_pose)
    if cond_mode in ("full",) + ACT:
        yaw = float(np.arctan2(start_pose[2], start_pose[3]))
        parts.append(occ_crop(occ, extent, start_pose[:2], yaw))
    if cond_mode in ACT:
        if action is None:
            raise ValueError(f"cond_mode {cond_mode} needs an action per segment")
        onehot = np.zeros(4, np.float32)
        onehot[ACTION_IDS[action]] = 1.0
        parts.append(onehot)
    if cond_mode == "full_action_head":
        # Target facing, relative to start. Priority: an explicit head_target (world yaw) --
        # e.g. the correct SEATED orientation, which the goal position cannot supply; else
        # command facing = direction of TRAVEL for a real move; else keep the current heading
        # (don't spin sitting/standing in place).
        start_xy = np.asarray(start_pose[:2], float)
        d = np.asarray(goal_world, float) - start_xy
        start_yaw = float(np.arctan2(start_pose[2], start_pose[3]))
        if head_target is not None:
            delta = float(head_target) - start_yaw
        elif np.linalg.norm(d) >= HEAD_MIN_DISP:
            delta = float(np.arctan2(d[1], d[0])) - start_yaw
        else:
            delta = 0.0
        parts.append(np.array([np.sin(delta), np.cos(delta)], np.float32))
    raw = np.concatenate(parts).astype(np.float32)
    return ((raw - cmean) / cstd).astype(np.float32)


def rollout(trans, net, clip_model, clip_mod, mean, std, ns, texts, goals,
            start_pose, prefix_pose, occ=None, extent=None, max_seg=None, actions=None,
            reorient=False, head_targets=None, scene_ctx=None, decode_iters=2,
            seg_headings=None, deskate_feet=False, deskate_floor=0.0):
    """Chain len(goals) segments. Returns list of per-segment dicts.

    actions: per-segment action name (walk/sit/stand up/lie), required when the model's
    cond_mode is full_action (Step 11) and ignored otherwise. Lets the caller command the
    action explicitly rather than leaving it to text -- the fix for the model treating
    every segment as navigation.

    reorient: if True, rotate each segment's START heading to face its goal before
    generating (walk-scale moves only). The model can only turn ~29 deg inside one short
    HUMANISE segment, so on free chains with sharp waypoint-to-waypoint turns the body
    lags its travel and "moonwalks" (RESULTS §11 heading limitation). Re-orienting makes
    travel "forward" every segment so the body faces where it walks. It does NOT break the
    seam: the prefix pose is heading-canonicalized (frame-independent), so this only rotates
    the body about its root, it does not teleport limbs. Interaction segments (short goal)
    keep their inherited facing so they don't spin in place."""
    cmean = np.array(ns["cond_mean"], np.float32)
    cstd = np.array(ns["cond_std"], np.float32)
    pose = np.asarray(start_pose, dtype=np.float32).copy()
    prefix = np.asarray(prefix_pose, dtype=np.float32).copy()
    segs = []
    with torch.no_grad():
        for k, (txt, goal) in enumerate(zip(texts, goals)):
            if max_seg and k >= max_seg:
                break
            act = actions[k] if actions is not None else None
            if reorient:
                dvec = np.asarray(goal, float) - pose[:2]
                if np.linalg.norm(dvec) >= HEAD_MIN_DISP:
                    ry = float(np.arctan2(dvec[1], dvec[0]))
                    pose = pose.copy(); pose[2], pose[3] = np.sin(ry), np.cos(ry)
            # Force a per-segment start heading (world yaw), overriding inherited/reorient. Used to
            # make a SIT segment face the PERCEIVED furniture facing (turn-to-sit) -- the seam stays
            # clean because the prefix pose is heading-canonicalized (RESULTS §11 reorient).
            if seg_headings is not None and seg_headings[k] is not None:
                h = float(seg_headings[k])
                pose = pose.copy(); pose[2], pose[3] = np.sin(h), np.cos(h)
            ht = head_targets[k] if head_targets is not None else None
            feat = clip_model.encode_text(
                clip_mod.tokenize([txt], truncate=True).to(DEV)).float()
            extra = build_cond(ns["cond_mode"], np.asarray(goal, float), pose,
                               prefix, occ, extent, cmean, cstd, action=act, head_target=ht)
            cond = torch.cat([feat, torch.from_numpy(extra).unsqueeze(0).to(DEV)], -1)
            tok = trans.sample(cond, if_categorial=False)
            if tok.numel() == 0:
                break
            if scene_ctx is not None:
                # Heightmap-conditioned (geometry-grounded) decode: resolve the circularity by
                # placing a first-pass decode, sampling the scene heightmap along it, re-decoding.
                from scene_decode import decode_with_heightmap  # lazy: only when scene-aware
                motion = decode_with_heightmap(net, tok, pose, scene_ctx, mf, std, mean,
                                               n_iters=decode_iters)
            else:
                motion = net.forward_decoder(tok)[0].cpu().numpy() * std + mean
            world = se2_place_full_body(motion.astype(np.float32), pose, mf)  # (T,22,3) Z-up
            if deskate_feet:
                # OUTPUT-ONLY foot-contact cleanup (src/foot_contact): removes the ~2.5x-GT
                # foot-skate the VQ-VAE injects (scripts/contact). Root/goal/bone-lengths are
                # preserved exactly. Applied to the placed world pose ONLY -- NOT to `local`/the
                # prefix handoff below, which must stay the on-manifold decoded pose (RESULTS §7
                # rule 3), exactly like blend_seam is display-only.
                from foot_contact import deskate
                world = deskate(world, floor=deskate_floor)

            local = mf.local_joint_positions(motion.astype(np.float32))
            segs.append({
                "world": world,
                "goal": np.asarray(goal, float),
                "start_pose": pose.copy(),
                "seam_err": float(np.linalg.norm(local[0] - prefix.reshape(22, 3), axis=-1).mean()),
                "goal_err": float(np.linalg.norm(world[-1, 0, :2] - np.asarray(goal, float))),
                "text": txt,
                "action": act,
            })
            # hand off: DECODED ending pose, and heading from the generated body
            end_xy = world[-1, 0, :2]
            end_yaw = yaw_from_joints(world[-1])
            pose = np.array([end_xy[0], end_xy[1], np.sin(end_yaw), np.cos(end_yaw)], np.float32)
            prefix = local[-1].ravel().astype(np.float32)
    return segs


def blend_seam(a_world, b_world, n=4):
    """Cosmetic 4-frame crossfade for DISPLAY ONLY (CLAUDE.md 2d). Never feed
    the blended pose back as conditioning -- it is off-manifold."""
    if a_world is None or len(b_world) < n or len(a_world) < n:
        return b_world
    out = b_world.copy()
    w = np.linspace(0, 1, n + 2)[1:-1][:, None, None]
    out[:n] = (1 - w) * a_world[-1][None] + w * b_world[:n]
    return out


def load_model(ckpt_dir):
    ns = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))
    tr = build_transformer(ns["clip_dim"])
    tr.load_state_dict(torch.load(os.path.join(ckpt_dir, "net_final.pth"),
                                  map_location="cpu")["trans"], strict=True)
    return tr.eval().to(DEV), ns
