#!/usr/bin/env python3
"""Full-mesh 3D demo: render generated motion as a skeleton figure INSIDE the real textured ScanNet
room mesh (not the top-down occupancy plot the other demos use). This is the watchable deliverable.

No SMPL body model is on disk and our pipeline is 22-joint (SMPL-X was discarded, CLAUDE.md 2a), so
the human is drawn as a balls-and-sticks skeleton (joint spheres + bone cylinders) rather than a
skinned mesh. The ROOM is the real `*_vh_clean_2.ply` with vertex colors, ceiling-clipped so an
oblique camera sees into it.

Two demo modes, both rendered the same way:
  interaction  walk -> sit -> stand -> walk on real furniture (step-11 action model + a sit-clip
               seed for a scene KNOWN to contain a seat; RESULTS §11). Retries seeds until a chain
               both SITS and STANDS (pelvis height), since composed yield is ~50%.
  navigation   a steered multi-waypoint walk (goalaug model + guided_seg collision-guided decoding,
               RESULTS §13) — shows the body threading furniture instead of walking through it.

World-frame joints (se2_place_full_body output) and the mesh share the Z-up ScanNet world frame, so
they overlay directly. Seams get the display-only crossfade (blend_seam, CLAUDE.md 2d).
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import clip
import numpy as np
import torch
import trimesh
from trimesh.creation import uv_sphere, cylinder
import pyrender

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
import bev_render  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import rollout, load_model, blend_seam  # noqa: E402
from demo_interaction import compose_goals_texts, object_phrase, pelvis_z  # noqa: E402
from demo_rollout import sample_waypoints  # noqa: E402
from collision_guided import run_chain  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
TALL = os.path.expanduser("~/wander_data/bev_tall_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# HumanML3D 22-joint kinematic chains -> bone (parent,child) pairs
CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
         [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
BONES = [(c[i], c[i + 1]) for c in CHAIN for i in range(len(c) - 1)]


def look_at(cam_pos, target, up=np.array([0, 0, 1.0])):
    f = target - cam_pos; f = f / np.linalg.norm(f)
    s = np.cross(f, up); s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    M = np.eye(4)
    M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = cam_pos
    return M


def skeleton_mesh(J, color=(245, 130, 32), joint_r=0.05, bone_r=0.032):
    """(22,3) joints -> one colored trimesh of joint spheres + bone cylinders."""
    geoms = []
    for j in range(len(J)):
        s = uv_sphere(radius=joint_r, count=[8, 8]); s.apply_translation(J[j]); geoms.append(s)
    for a, b in BONES:
        if np.linalg.norm(J[a] - J[b]) < 1e-4:
            continue
        geoms.append(cylinder(radius=bone_r, segment=np.array([J[a], J[b]]), sections=10))
    m = trimesh.util.concatenate(geoms)
    m.visual.vertex_colors = np.tile(np.array([*color, 255], np.uint8), (len(m.vertices), 1))
    return m


def clip_ceiling(mesh, cutoff_m):
    """Drop faces whose lowest vertex is > cutoff_m above the floor, so an oblique camera sees in."""
    floor_z = mesh.bounds[0, 2]
    fz = mesh.vertices[mesh.faces][:, :, 2]
    keep = fz.min(axis=1) <= floor_z + cutoff_m
    c = mesh.copy(); c.faces = mesh.faces[keep]; c.remove_unreferenced_vertices()
    return c, floor_z


def write_mp4(frames, path, fps):
    H, W = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f[:, :, :3], np.uint8).tobytes())
    p.stdin.close(); p.wait()


def _smooth(path, win=11):
    """Moving-average smooth an (N,D) path so a follow-camera target doesn't jitter."""
    if len(path) < 3:
        return path
    k = min(win, len(path) | 1)  # odd, <= len
    pad = k // 2
    p = np.pad(path, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(p[:, d], ker, mode="valid") for d in range(path.shape[1])], 1)


def render_in_mesh(scene_id, world_frames, out_path, fps=20, cam_az=None, cam_elev=52,
                   ceiling=2.0, res=(960, 720), stride=1, orbit=20.0, follow=True,
                   follow_dist=3.4):
    """world_frames: (N,22,3) Z-up world joints. Renders each into the ceiling-clipped room mesh.

    follow=True: a tracking camera locked at a fixed azimuth/elevation a short distance behind the
    (smoothed) pelvis, so the figure stays large and centered and the room scrolls past -- far more
    legible than one fixed wide shot where a multi-metre walk shrinks the body to a few pixels.
    follow=False: one fixed shot framing the whole path, with a gentle `orbit` sweep."""
    mesh = bev_render._load_scene_mesh(scene_id)
    room, floor_z = clip_ceiling(mesh, ceiling)
    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.55, 0.55, 0.55])
    scene.add(pyrender.Mesh.from_trimesh(room, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.radians(48))
    cam_node = scene.add(cam, pose=np.eye(4))
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.5)
    light_node = scene.add(light, pose=np.eye(4))

    frames_j = world_frames[::stride]
    root = frames_j[:, 0, :]                       # pelvis path
    if cam_az is None:                             # view along the dominant travel direction, from the side
        disp = root[-1, :2] - root[0, :2]
        travel = float(np.degrees(np.arctan2(disp[1], disp[0]))) if np.linalg.norm(disp) > 0.5 else 45.0
        cam_az = travel - 90.0                     # 90 deg to the side of travel
    targets = _smooth(root[:, :2], 15) if follow else None
    span = float(np.linalg.norm(frames_j[:, :, :2].reshape(-1, 2).ptp(0)))
    fixed_target = np.array([root[:, 0].mean(), root[:, 1].mean(), floor_z + 0.9])
    fixed_dist = max(3.2, span * 0.9 + 2.6)

    W, H = res
    r = pyrender.OffscreenRenderer(W, H)
    out = []
    try:
        for i, fr in enumerate(frames_j):
            if follow:
                target = np.array([targets[i, 0], targets[i, 1], floor_z + 0.9])
                az, dist = np.radians(cam_az), follow_dist
            else:
                target = fixed_target
                az = np.radians(cam_az + orbit * np.sin(np.pi * i / max(1, len(frames_j) - 1)))
                dist = fixed_dist
            el = np.radians(cam_elev)
            cam_pos = target + np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el),
                                         np.sin(el)]) * dist
            pose = look_at(cam_pos, target)
            scene.set_pose(cam_node, pose); scene.set_pose(light_node, pose)
            hm = pyrender.Mesh.from_trimesh(skeleton_mesh(fr), smooth=False)
            hn = scene.add(hm)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            out.append(color.copy())
            scene.remove_node(hn)
    finally:
        r.delete()
    write_mp4(out, out_path, fps)
    return len(out)


def stitch(segs, blend_n):
    worlds, prev = [], None
    for s in segs:
        w = blend_seam(prev, s["world"], n=blend_n); worlds.append(w); prev = w
    return np.concatenate(worlds)


def load_common():
    net = load_vqvae(ckpt_path=ARGS.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(ARGS.ckpt)
    return net, mean, std, cmodel, trans, ns


def scene_assets(rec, idx):
    fb, ft = os.path.join(BEV, f"{rec.scene}.npz"), os.path.join(TALL, f"{rec.scene}.npz")
    if not (os.path.exists(fb) and os.path.exists(ft)):
        return None
    p = f"{bev_render.SCANNET_ROOT}/{rec.scene}/{rec.scene}_vh_clean_2.ply"
    if not os.path.exists(p):
        return None
    zb, zt = np.load(fb), np.load(ft)
    return zb["occ"].astype(np.float32), zb["extent"], zt["occ"].astype(np.float32)


def gen_interaction(net, mean, std, cmodel, trans, ns):
    """Retry sit seeds until a chain SITS and STANDS. Returns (segs, rec, actions) or None."""
    flat = build_flat_join()
    seeds = [i for i, p in enumerate(flat) if p["action"] == "sit"]
    rng = np.random.RandomState(ARGS.seed); rng.shuffle(seeds)
    tried = 0
    for idx in seeds:
        if tried >= ARGS.n_try:
            break
        rec = get_record(int(idx))
        if ARGS.scene and rec.scene != ARGS.scene:
            continue
        a = scene_assets(rec, idx)
        if a is None:
            continue
        occ, extent, tall = a
        cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
        try:
            d0, *_ = mf.humanise_positions_to_263(cm)
        except Exception:
            continue
        if d0.shape[0] < 8:
            continue
        _, xy, _, sincos = compute_track2(rec)
        sit_xy = xy[-1].astype(np.float32)
        prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel()
        # synthesize a real walk-up start (recipe from IN_FLIGHT: start-dist 3.0, front 0.3)
        sp = sample_waypoints(occ, extent, sit_xy, 1, min_step=max(0.6, ARGS.start_dist - 0.5),
                              rng=rng, max_step=ARGS.start_dist + 0.5)
        if not sp:
            continue
        s_xy = np.asarray(sp[0], float); d = sit_xy - s_xy
        yaw = float(np.arctan2(d[1], d[0]))
        start_pose = np.array([s_xy[0], s_xy[1], np.sin(yaw), np.cos(yaw)], np.float32)
        goals, nwi = compose_goals_texts(start_pose[:2], sit_xy, occ, extent, rng,
                                         front=ARGS.front, stand=0.5, away=1.3)
        obj = object_phrase(rec.utterance)
        texts = [f"walk to {obj}"] * nwi + [rec.utterance, f"stand up from {obj}", "walk to the door"]
        actions = ["walk"] * nwi + ["sit", "stand up", "walk"]
        tried += 1
        segs = rollout(trans, net, cmodel, clip, mean, std, ns, texts, goals, start_pose, prefix,
                       occ, extent,
                       actions=actions if ns["cond_mode"] in ("full_action", "full_action_head") else None,
                       reorient=True)
        if len(segs) < len(goals):
            continue
        zs = [pelvis_z(s) for s in segs]
        sat = float(zs[nwi][-1]) < 0.7 and float(zs[nwi + 1][-1]) > 0.8
        print(f"  try {tried} {rec.scene} {obj:20s} SAT&STOOD={sat}", flush=True)
        if sat:
            return segs, rec, obj
    return None


def gen_navigation(net, mean, std, cmodel, trans, ns):
    flat = build_flat_join()
    walk = [i for i, p in enumerate(flat) if p["action"] == "walk"]
    rng = np.random.RandomState(ARGS.seed); rng.shuffle(walk)
    tried = 0
    best = None
    for idx in walk:
        if tried >= ARGS.n_try:
            break
        rec = get_record(int(idx))
        if ARGS.scene and rec.scene != ARGS.scene:
            continue
        a = scene_assets(rec, idx)
        if a is None:
            continue
        occ, extent, tall = a
        cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
        try:
            d0, *_ = mf.humanise_positions_to_263(cm)
        except Exception:
            continue
        if d0.shape[0] < 8:
            continue
        _, xy, _, sincos = compute_track2(rec)
        start_pose = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
        prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel()
        wps = sample_waypoints(occ, extent, start_pose[:2], ARGS.n_segments, 0.6, rng, max_step=1.2)
        if wps is None:
            continue
        texts = ["walk to the target"] * ARGS.n_segments
        tried += 1
        segs = run_chain(trans, net, cmodel, mean, std, ns, texts, wps, start_pose, prefix,
                         occ, extent, tall, True, "guided_seg", 8, 10.0, rng)
        if len(segs) < ARGS.n_segments:
            continue
        path = np.concatenate([s["world"][:, 0, :2] for s in segs])
        from collision_guided import path_collision
        coll = path_collision(path, tall, extent)
        print(f"  try {tried} {rec.scene} coll={coll*100:.1f}%", flush=True)
        # prefer a scene where steering matters (some obstacle contact avoided) but still low
        if best is None or (0.2 < coll < 3.0 and coll < best[0]):
            best = (coll, segs, rec)
        if best and best[0] < 1.0 and tried >= 4:
            break
    if best is None:
        return None
    return best[1], best[2], "waypoints"


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["interaction", "navigation"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default=None, help="restrict to a specific scene id")
    ap.add_argument("--n-try", type=int, default=30)
    ap.add_argument("--n-segments", type=int, default=6, help="navigation waypoint count")
    ap.add_argument("--start-dist", type=float, default=3.0, help="interaction walk-up distance")
    ap.add_argument("--front", type=float, default=0.3)
    ap.add_argument("--blend-n", type=int, default=6)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--cam-az", type=float, default=None)
    ap.add_argument("--cam-elev", type=float, default=52)
    ap.add_argument("--ceiling", type=float, default=2.0)
    ap.add_argument("--orbit", type=float, default=20.0, help="degrees of gentle camera sweep (fixed cam)")
    ap.add_argument("--no-follow", dest="follow", action="store_false", default=True,
                    help="use one fixed wide shot instead of the tracking camera")
    ap.add_argument("--follow-dist", type=float, default=3.4, help="tracking-camera distance (m)")
    ap.add_argument("--seed", type=int, default=0)
    ARGS = ap.parse_args()
    os.makedirs(ARGS.out, exist_ok=True)
    torch.manual_seed(ARGS.seed)

    net, mean, std, cmodel, trans, ns = load_common()
    print(f"mode={ARGS.mode} cond_mode={ns['cond_mode']}\n", flush=True)
    got = gen_interaction(*(net, mean, std, cmodel, trans, ns)) if ARGS.mode == "interaction" \
        else gen_navigation(*(net, mean, std, cmodel, trans, ns))
    if got is None:
        print("no suitable chain produced"); return
    segs, rec, label = got
    allw = stitch(segs, ARGS.blend_n)
    out_path = os.path.join(ARGS.out, f"mesh_{ARGS.mode}_{rec.scene}_{ARGS.seed}.mp4")
    n = render_in_mesh(rec.scene, allw, out_path, fps=ARGS.fps, cam_az=ARGS.cam_az,
                       cam_elev=ARGS.cam_elev, ceiling=ARGS.ceiling, orbit=ARGS.orbit,
                       follow=ARGS.follow, follow_dist=ARGS.follow_dist)
    print(f"\nrendered {n} frames -> {out_path}  ({rec.scene}, {label})", flush=True)


if __name__ == "__main__":
    main()
