#!/usr/bin/env python3
"""Furniture-anchor detection + scene annotation — the VLM's grounding input (build-order step 13).

The planner does NOT ask the VLM to regress raw world coordinates (VLMs are poor at that). Instead we
extract candidate targets from GEOMETRY and let the VLM do what it is good at: pick WHICH one and WHAT
action. A "low furniture" cell is occupied in the 0.12 m map but FREE in the 0.9 m map (RESULTS §8) —
exactly sofas, beds, low tables, chair seats, i.e. the sit-able surfaces — while walls/doors (tall)
and open floor drop out. Connected components of that mask are the anchors. Each anchor's world
centroid is computed from geometry; the VLM only ever references it by number.

annotate() overlays numbered markers on the top-down photographic BEV so the VLM sees the room AND a
stable id per target. pixel<->world uses the shared BEV extent (same convention as demo_rollout).
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
BEV = os.path.expanduser("~/wander_data/bev_cache")
TALL = os.path.expanduser("~/wander_data/bev_tall_cache")


def px_to_world(rc, extent, shape):
    xmin, xmax, ymin, ymax = extent
    H, W = shape
    r, c = rc
    return np.array([xmin + (c + 0.5) / W * (xmax - xmin),
                     ymax - (r + 0.5) / H * (ymax - ymin)])


def world_to_px(xy, extent, shape):
    xmin, xmax, ymin, ymax = extent
    H, W = shape
    c = int(np.clip((xy[0] - xmin) / (xmax - xmin) * W, 0, W - 1))
    r = int(np.clip((ymax - xy[1]) / (ymax - ymin) * H, 0, H - 1))
    return r, c


def load_scene_maps(scene_id):
    """Returns (rgb HxWx3 uint8, occ 0.12m bool, tall 0.9m bool, extent). tall is resized to occ."""
    zb = np.load(os.path.join(BEV, f"{scene_id}.npz"))
    zt = np.load(os.path.join(TALL, f"{scene_id}.npz"))
    rgb = zb["rgb"]; occ = zb["occ"].astype(bool); extent = zb["extent"]
    tall = zt["occ"].astype(bool)
    if tall.shape != occ.shape:  # align if the two caches rendered at different resolutions
        ti = np.array(Image.fromarray(tall.astype(np.uint8) * 255).resize(
            (occ.shape[1], occ.shape[0]), Image.NEAREST)) > 127
        tall = ti
    return rgb, occ, tall, extent


def detect_anchors(occ, tall, extent, min_area_m2=0.12, max_area_m2=6.0, max_anchors=10):
    """Low-furniture connected components -> anchors (id, world xy, pixel rc, area). Sorted by area."""
    from scipy import ndimage
    low = occ & (~tall)
    low = ndimage.binary_opening(low, np.ones((3, 3)))  # drop 1-px scan noise
    lbl, n = ndimage.label(low)
    H, W = occ.shape
    xmin, xmax, ymin, ymax = extent
    cell_m2 = (xmax - xmin) / W * (ymax - ymin) / H
    cands = []
    for i in range(1, n + 1):
        mask = lbl == i
        area = int(mask.sum()) * cell_m2
        if area < min_area_m2 or area > max_area_m2:
            continue
        rr, cc = np.nonzero(mask)
        rc = (rr.mean(), cc.mean())
        cands.append(dict(rc=rc, xy=px_to_world(rc, extent, occ.shape), area=area,
                          npix=int(mask.sum())))
    cands.sort(key=lambda a: -a["area"])
    cands = cands[:max_anchors]
    for k, a in enumerate(cands):
        a["id"] = k + 1
    return cands


def annotate(rgb, extent, anchors, out_path, start_xy=None):
    """Draw numbered markers (and optional start pose) on the BEV; save PNG for the VLM."""
    img = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
    up = 1.4
    img = img.resize((int(img.width * up), int(img.height * up)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    shape = (rgb.shape[0], rgb.shape[1])
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    for a in anchors:
        r, c = a["rc"]; px, py = c * up, r * up
        rad = 20
        d.ellipse([px - rad, py - rad, px + rad, py + rad], outline=(255, 40, 40), width=5)
        d.text((px - 8, py - 15), str(a["id"]), fill=(255, 255, 0), font=font,
               stroke_width=2, stroke_fill=(0, 0, 0))
    if start_xy is not None:
        r, c = world_to_px(start_xy, extent, shape)
        px, py = c * up, r * up
        d.ellipse([px - 14, py - 14, px + 14, py + 14], fill=(30, 120, 255))
        d.text((px + 16, py - 10), "START", fill=(30, 120, 255), font=font,
               stroke_width=2, stroke_fill=(255, 255, 255))
    img.save(out_path)
    return out_path


def anchor_legend(anchors):
    """Human/VLM-readable list of anchors with world coords + rough size hint."""
    lines = []
    for a in anchors:
        size = "large" if a["area"] > 1.5 else ("small" if a["area"] < 0.4 else "medium")
        lines.append(f"  #{a['id']}: world ({a['xy'][0]:.2f}, {a['xy'][1]:.2f}) m, "
                     f"{size} low-furniture footprint ({a['area']:.2f} m^2)")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="/home/dsp52026/.claude/jobs/72353c49/tmp/anchors.png")
    args = ap.parse_args()
    rgb, occ, tall, extent = load_scene_maps(args.scene)
    anchors = detect_anchors(occ, tall, extent)
    print(f"{len(anchors)} anchors in {args.scene}:")
    print(anchor_legend(anchors))
    annotate(rgb, extent, anchors, args.out)
    print(f"annotated -> {args.out}")
