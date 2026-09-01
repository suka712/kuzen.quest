#!/usr/bin/env python3
"""Local-MLLM planner (build-order step 13): instruction + annotated scene image -> ordered motion
plan. Uses ollama `qwen3.5:27b` (vision) — the same local model as the orientation probe (no cloud
API, CLAUDE.md deliverable). The VLM does the SPATIAL GROUNDING by choosing WHICH numbered anchor
each segment targets; geometry (scene_anchors) turns the choice into a world coordinate, so the motion
model still never guesses a location (CLAUDE.md 2c).

Output schema (forced JSON): {"segments": [{"action","target","why"}, ...]}
  action  one of: walk | sit | stand up | lie   (exactly the motion model's 4-way action one-hot)
  target  an anchor id (int from the legend) OR "away" (leave / open floor away from furniture)
  why     short justification (for the demo caption + debugging the grounding)
"""
import base64
import json
import os
import sys
import urllib.request

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts/planner"))
import numpy as np
from scene_anchors import (load_scene_maps, detect_anchors, annotate, anchor_legend)  # noqa: E402

OLLAMA = "http://localhost:11434/api/generate"
MODEL = os.environ.get("WANDER_PLANNER_MODEL", "qwen3.5:27b")

SYSTEM = """You are the motion planner for an indoor humanoid. You are given:
- a TOP-DOWN photo of a room, with candidate targets marked as numbered red circles, and the
  person's START marked in blue;
- a natural-language instruction;
- a legend listing each numbered target's world (x,y) position and rough size.

Decompose the instruction into an ordered sequence of body-motion SEGMENTS the person performs,
starting from START. Rules:
- Each segment has an "action" (one of: "walk", "sit", "stand up", "lie") and a "target".
- "target" is either a target NUMBER from the legend, or the string "away" (meaning walk to open
  floor / leave the room).
- To sit or lie on a piece of furniture the person must FIRST "walk" to that same target, THEN
  "sit"/"lie" on it (two segments, same target number).
- After a "sit" or "lie", the person must "stand up" (same target) before walking elsewhere.
- Pick the target whose position and appearance best matches the instruction (e.g. a couch/sofa for
  "sit and rest", a bed for "lie down"). Sub-goals like "get a coffee" become a "walk" to the
  nearest plausible target, not a depicted grasp.
- Keep it to 2-6 segments. Use ONLY target numbers that appear in the legend.
Respond with ONLY JSON: {"segments":[{"action":..,"target":..,"why":..}, ...]}"""


def _b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def call_qwen(image_path, instruction, legend, timeout=300):
    prompt = (f"{SYSTEM}\n\nINSTRUCTION: {instruction}\n\nTARGET LEGEND:\n{legend}\n\n"
              "Return the JSON plan now.")
    payload = {"model": MODEL, "prompt": prompt, "images": [_b64(image_path)],
               "stream": False, "think": False, "format": "json",
               "options": {"temperature": 0, "num_predict": 512}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r.get("response", "").strip()


def unload(model=MODEL, timeout=20):
    """Evict the VLM from GPU memory (ollama keep_alive:0). The 27B holds ~18 GB; the motion model
    that runs next OOMs (CUBLAS_STATUS_NOT_INITIALIZED) unless the planner frees the GPU first."""
    try:
        payload = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(OLLAMA, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=timeout).read()
    except Exception:
        pass


def parse_plan(resp, anchors):
    """Parse the VLM JSON; resolve each target to a world (x,y). Returns list of resolved segments."""
    txt = resp
    if "```" in txt:  # strip code fences if present
        txt = txt.split("```")[1].replace("json", "", 1).strip()
    obj = json.loads(txt)
    segs = obj["segments"] if isinstance(obj, dict) else obj
    by_id = {a["id"]: a for a in anchors}
    out = []
    ACT = {"walk": "walk", "sit": "sit", "stand up": "stand up", "standup": "stand up",
           "stand": "stand up", "lie": "lie", "lie down": "lie"}
    for s in segs:
        act = ACT.get(str(s.get("action", "")).strip().lower())
        if act is None:
            continue
        tgt = s.get("target")
        if isinstance(tgt, str) and tgt.strip().lower() in ("away", "exit", "door", "leave"):
            resolved = {"action": act, "target": "away", "xy": None, "why": s.get("why", "")}
        else:
            try:
                aid = int(tgt)
            except (TypeError, ValueError):
                continue
            if aid not in by_id:
                continue
            resolved = {"action": act, "target": aid, "xy": by_id[aid]["xy"].astype(float),
                        "why": s.get("why", "")}
        out.append(resolved)
    return out


def make_plan(scene_id, instruction, start_xy, out_img, timeout=300):
    """Full planner: detect anchors, annotate, query the VLM, parse. Returns (plan, anchors, raw)."""
    rgb, occ, tall, extent = load_scene_maps(scene_id)
    anchors = detect_anchors(occ, tall, extent)
    annotate(rgb, extent, anchors, out_img, start_xy=start_xy)
    legend = anchor_legend(anchors)
    raw = call_qwen(out_img, instruction, legend, timeout=timeout)
    plan = parse_plan(raw, anchors)
    return plan, anchors, raw


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--start", default="0,0", help="start x,y in world meters")
    ap.add_argument("--out-img", default="/home/dsp52026/.claude/jobs/72353c49/tmp/plan_scene.png")
    args = ap.parse_args()
    sx = np.array([float(v) for v in args.start.split(",")])
    plan, anchors, raw = make_plan(args.scene, args.instruction, sx, args.out_img)
    print("RAW VLM RESPONSE:\n", raw, "\n")
    print("RESOLVED PLAN:")
    for i, s in enumerate(plan):
        loc = "open floor (away)" if s["target"] == "away" else \
              f"#{s['target']} @ ({s['xy'][0]:.2f},{s['xy'][1]:.2f})"
        print(f"  {i+1}. {s['action']:9s} -> {loc}   [{s['why']}]")
