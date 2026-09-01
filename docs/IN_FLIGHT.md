# In flight — read this if you are picking up cold

Volatile state that is NOT captured by RESULTS.md: what is running, where things live on the
boxes, and the next concrete action. **Update or delete this file when its work lands.**

Last updated: 2026-09-02. Steps 1-12 done. Best interaction model
`~/wander_data/step11/checkpoints/action` (`cond_mode=full_action`); best navigation model
`~/wander_data/step10/checkpoints/goalaug` (`cond_mode=full`); finetuned VQ-VAE
`~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth`.

**STEP 12 DONE (2026-09-02) — collision-guided decoding works. RESULTS §13.**
`scripts/chaining/collision_guided.py` adds inference-time scene steering (no training). On 20×6
chained rollouts, two seeds: greedy collides 2.06%/3.63% (≥ a straight line — the model doesn't
steer); **guided_seg (per-segment best-of-N on `goal_err + 10·collision`) cuts it to 0.69%/2.64%,
BELOW the straight-line oracle (1.57%/3.93%), while IMPROVING goal error to 0.09–0.11 m.** Rejection
sampling (`reject_chain`) is the floor (0.85%/2.19%) but degrades goal error. Figure:
`~/wander_data/step12_fig/cg_compare_scene0001_00.png` (greedy walks through an obstacle 16.3% →
guided 0.0%). This is the last SWAPPABLE contribution; navigation+chaining+grounding+steering is now
a complete story. **Next: step 13 (Qwen JSON end-to-end demo) and/or the writeup.**

**STEP 13 DONE (2026-09-02) — end-to-end VLM pipeline works. RESULTS §14.** `scripts/planner/`:
`scene_anchors.py` (low-furniture connected components -> numbered anchors on the BEV), `qwen_plan.py`
(ollama `qwen3.5:27b` vision -> JSON `{action,target}` plan; VLM grounds WHICH anchor, geometry gives
the xy), `demo_end2end.py` (expand -> guided_seg-steered rollout with the step-11 action model ->
full-mesh render). Verified on scene0151 "sit and relax on the couch…": qwen grounded the couch to
anchor #1, chain SAT (pelvis 0.56 m) and STOOD (0.95 m). Demo + plan image shipped to
`mcx:/Users/khiem/Documents/wander-output/step13/`. VLM plan ~80 s (27B). The sit still lands at the
seat EDGE (the §11 narrowness carries through the VLM plan unchanged). **Next: step 14 (benchmark +
FID) or demo polish (skinned body mesh, sit-placement).**

**FULL-MESH 3D DEMOS (2026-09-02).** `scripts/chaining/render_mesh_demo.py` renders generated motion
as a balls-and-sticks skeleton (no SMPL on disk) INSIDE the real textured `*_vh_clean_2.ply` room
mesh (ceiling-clipped), with a tracking follow-camera. Two modes: `--mode interaction` (walk→sit→
stand→walk, step-11 action model, retries seeds for SAT&STOOD) and `--mode navigation` (guided_seg
steering, goalaug). Validated the render path with a GT-joints oracle first (skeleton sits correctly
on the couch). Outputs in `~/wander_data/step12_mesh_demo/`; the couch-sit + office-navigation clips
+ the cg_compare figure were shipped to the Mac at `mcx:/Users/khiem/Documents/wander-output/step12/`
(scp via sshpass, key auth not set up). NOTE the sit still lands at the seat EDGE/corner, not squarely
(the documented sit-placement narrowness), so interaction clips are watchable but not crisp; couches
read better than armchairs.

**Two big investigations closed since (both documented, do not redo):**
- **Geometry-grounded tokenizer (SceMoS port): explored → MARGINAL, not worth shipping. RESULTS §12.**
  Both the frozen-decoder and the full unfreeze+shift-consistency cascade were built & measured; the
  contact-height gain is a wash and it costs broad motion quality. Keep the scene-blind tokenizer.
- **Sit orientation is a DATA limitation, not tunable → do it in the PLANNER.** HUMANISE bakes in
  approach≈sit-facing (median 5°), so the motion model can't learn facing; but a geometry perceiver
  (`probe_furniture_orientation.py::perceive_facing`, "away from the backrest") gets ~30–36° median on
  clear-fronted furniture (a VLM, ollama qwen3.5:27b vision, is comparable and NOT better). See the
  "Open limitation — sit orientation" section below for the full numbers.

**→ NEXT ACTION (in progress): orientation-driven "true sit" demo.** Wire the geometry perceiver into
the demo so the APPROACH is planned from perceived furniture facing (not GT), curated to clear-fronted
furniture (sofa/chair/bed). Reuses `scripts/chaining/{demo_interaction,rollout}.py` + the step-11 model.
This is a legitimate planner-side fix (no GT peek, no motion-model tuning). **[DONE — narrow/low-yield,
see the sit-orientation section. Step 12 is now also DONE, see the banner above.]**

---

## Nothing is training right now. (An eval may be running — check.)

```
ssh train-3090 'pgrep -af "[t]rain_probe.py|[d]emo_interaction|[e]val_"; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

**Wait-loop gotcha.** `while pgrep -f foo.py; do sleep 60; done` never exits — the loop's own
command line contains "foo.py" so pgrep matches itself. Use `pgrep -f "[f]oo.py"` or poll a
log sentinel (`grep -q '^saved ' log`).

## Step-11 result, reproduce commands

Best model `~/wander_data/step11/checkpoints/action` (`cond_mode=full_action`, goal-aug 0.5
walk-only, walk-prefix-aug 0.5). Recipe:
`WANDER_TRACK1_PROBE_ROOT=~/wander_data/step11 train_probe.py --conditioned --cond-mode
full_action --iters 20000 --lr 1e-4 --goal-aug 0.5 --walk-prefix-aug 0.5 --tokens-dir
~/wander_data/step10/tokens --out-name action`.

- Capability (pelvis height, NOT goal error which is z-blind): `eval_sit_capability.py --ckpts
  ~/wander_data/step11/checkpoints/action --vqvae-ckpt <ft-vqvae> --n 60 --actions sit "stand up"
  --prefix-mode walk` → sit **85%** (was 0%); `--prefix-mode own` → **83%** (was 42%).
- Demo: `demo_interaction.py --ckpt ~/wander_data/step11/checkpoints/action --vqvae-ckpt
  <ft-vqvae> --out <dir> --seed-action sit --n-demos 10 --start-dist 3.0 --front 0.3` →
  **5/10** chains SAT and STOOD. The walk-up AUTO-SPLITS into ≤1.1 m hops (a single 3 m walk
  undershoots and leaves the sit goal too long → the model walks instead of sitting).
  `--seed-action lie` for a lie demo (untried — worth a run).

## Next: step 13 (end-to-end demo) or the writeup

Done-criteria 4/5 are met AND step 12 is done, so both demo gates are cleared. Remaining
build-order items:
- **Step 12 — collision-guided decoding — DONE (2026-09-02, RESULTS §13).** `collision_guided.py`;
  guided_seg (w=10, n_cand=8) beats the straight-line oracle on both seeds while improving goal
  error. Reproduce: `collision_guided.py --ckpt ~/wander_data/step10/checkpoints/goalaug
  --vqvae-ckpt <ft-vqvae> --out <dir> --n-rollouts 20 --n-cand 8 --coll-weight 10 --seed {0,1}`.
  Figure: `render_cg_compare.py`.
- **Demo polish** (optional): close the composed SAT gap (50% → higher) with end-of-walk
  prefixes in `--walk-prefix-aug` (currently mid-stride only) or a higher aug probability; a
  `lie` demo; nicer camera. See RESULTS §11 "honest gap".
- **Step 13 — Qwen JSON** end-to-end now emits `{action, goal_coord}` per segment, which maps
  directly onto `demo_interaction`'s (action, goal) segments — the action one-hot is exactly the
  MLLM's `action` field.

## Next direction (when fresh) — geometry-grounded tokenizer, from SceMoS

The genuine "make the MODEL scene-aware for interaction" upgrade, learned from SceMoS (CVPR 2026,
arXiv 2602.20476, TRUMANS): put scene geometry in the **TOKENIZER**, not just the transformer.
- SceMoS's VQ-VAE decoder takes `(token, local heightmap)` — heightmap ±0.6 m in the body frame,
  32×32, recomputed each step, plain concatenation (beat FiLM/cross-attn), with a foot-contact
  reconstruction loss. Result: contact-correct motion (Contact 0.98, low penetration).
- Ours: tokenizer is scene-BLIND (decodes a canonical clip, SE(2)-placed); scene is only a
  transformer-side occupancy FOOTPRINT (height/orientation blind). That is why our interactions
  can't be contact-correct and the occupancy signal is weakly used.
- Sketch of the port: retrain the VQ-VAE with a local heightmap input to the decoder + a contact
  loss.

**BLOCKER RESOLVED 2026-08-28 — HUMANISE DOES give usable per-frame surface geometry.**
`scripts/scene_tokenizer/probe_heightmap.py` (no model/GPU/263 path — just `compute_track2` for the
world track + `bev_render._load_scene_mesh`, sampled with a cKDTree) builds the SceMoS ±0.6 m /
32×32 body-frame heightmap under the root. n=30/action, the ORACLE (support-surface height under
the body vs pelvis height, both above floor):

| action | pelvis_h | support_h | clearance (pelvis−support) | fill% |
|---|---|---|---|---|
| sit | 0.65 | 0.70 (raised seat) | −0.05 | 94 |
| lie | 0.28 | 0.29 (raised bed/mat) | −0.01 | 98 |
| stand up | 0.79 | 0.69 (at furniture) | 0.10 | 90 |
| walk | 1.00 | 0.31 (floor/passing furniture) | **0.69** | 90 |

The discriminator is **clearance**: an interacting body rests ON the surface (~0), a walking body
is high ABOVE it (0.69). Montages (`~/wander_data/scene_tokenizer_probe/heightmap_*.png`) show
coherent seats/beds, not noise. Fill 90–98% at 32×32 → the mesh is dense enough; only 2–10% of
cells need nearest-neighbor densifying. **The geometry-grounded tokenizer is viable; proceed to
the build.** One sampling caveat for the real extractor: naive max-Z-in-column catches walls /
overhead beside the target (one sit clip read 2.5 m) — either keep it (legit "obstacle here"
signal) or clip to a support-surface definition; decide in the extractor, not now.

### OUTCOME (2026-08-28 frozen; 2026-09-01 unfrozen) — BUILT & FULLY MEASURED. Mechanism works; BOTH approaches are marginal on our data. Full write-up: RESULTS §12. Recommendation: keep the scene-blind tokenizer; do not build further on this without new evidence.

**The full unfreeze retrain (user-authorized) is DONE and also marginal.** Unfroze encoder+quantizer
with a shift-consistency loss to make tokens height-agnostic (invariance plateaued 0.72 at
consist-weight 0.25), re-extracted all tokens (`~/wander_data/scene_tokenizer/tokens`, crop = 4×old
token length — NOT crop_to_multiple), retrained the transformer (`checkpoints/action_scene`, 97.2%
acc). Generated sits now track seat height (corr 0.1→0.6 vs the old fixed-nominal pipeline) BUT add
a ~7 cm overshoot (GT pelvis sits 0.156 m above the seat; heightmap decode lands at 0.229) so
absolute contact is a wash, and it cost broad MPJPE regression (sit 48→74). Tokenizer:
`checkpoints/scene_vqvae_unfrozen/net_iter007500.pth`. Gate: `gate_height_agnostic.py`. See RESULTS §12.

Original frozen finding (kept for the record):

The whole port was built and characterized (all code under `src/scene_*.py`, `src/contact_loss.py`,
`scripts/scene_tokenizer/*`). What was learned, in order:
1. **Extraction (done, validated).** `extract_heightmaps.py` → `~/wander_data/motion_data/HUMANISE_heightmap_cache`
   (19,648 clips, (T,32,32) f16, offset-0 aligned to the 263 cache). **Vertical-reference bug caught by an
   oracle:** the heightmap must be CLIP-FLOOR referenced (`scene_heightmap.to_clip_frame`), not scene-floor —
   a lie-on-bed body else reads 0.8 m "under" the bed (GT penetration 400–770 mm → ~0 after the fix).
2. **Decoder + freeze insight (done).** `SceneVQVAE` fuses a per-frame heightmap into the (pretrained) decoder,
   identity-init so recon == base at iter 0. **Freezing the encoder+quantizer keeps tokens BIT-IDENTICAL**
   (verified) → step10/tokens + the step-11 transformer are reused, NO re-tokenize/retrain. This collapsed
   the planned 5-stage cascade — and is also why the cheap version is limited (below).
3. **Redundancy trap #1 (reconstruction).** With plain recon+penetration the decoder IGNORES the heightmap
   (`follow_ratio` 0.00 at 2.5k) — the token already determines contact height. Fixed with **vertical-shift
   augmentation** (`train_scene_vqvae --height-aug`: encode the original clip, shift the heightmap by Δ, require
   the body to shift by Δ). follow → **~1.0**. Model: `~/wander_data/scene_tokenizer/checkpoints/scene_vqvae/net_iter020000.pth`.
4. **Redundancy trap #2 (generation) — THE BLOCKER, not solved.** `follow=1.0` is the RECONSTRUCTION regime
   (token consistent with the hm). At GENERATION the transformer emits a *generic* sit token that already
   encodes a nominal contact height (~0.6 m), and with the encoder FROZEN that token dominates the heightmap.
   Measured (`eval_contact_demo.py`, iterate-converged): generated seated pelvis is ~constant **~0.6 m**
   regardless of the real seat (0.15–0.4 m overshoot on low seats), corr(seat) only +0.2. It tracks well ONLY
   on tall seats (>0.9 m) where the scene-blind undershoot is even worse. So aggregate contact does NOT
   cleanly improve.

**Root cause & the real fix.** Frozen tokens carry the contact height that competes with the heightmap. The
proper SceMoS result needs the ENCODER trained heightmap-aware so tokens DON'T encode absolute contact height
(forcing the decoder to use the hm) → that means UNFREEZING encoder+quantizer, re-extracting tokens, and
retraining the transformer (the full cascade, ~hours). **Decision deferred to the user:** the scene-blind
tokenizer already meets done-criteria 4/5 (watchable sits), so contact-refinement is a nice-to-have whose
payoff — even with the full retrain — is now uncertain. Do not launch the unfreeze retrain without deciding
it's worth it. Two cheaper things to try FIRST if pursuing: (a) a **support-surface** heightmap (max-Z catches
backrests, inflating the seat ~0.15 m — a lower-percentile/under-body definition may cut the overshoot);
(b) generation-regime augmentation (token-dropout of the height channels so the decoder must rely on the hm).
- Inference circularity WAS solved: `scene_decode.decode_with_heightmap` (scene-blind first pass → SE(2) place
  → sample heightmaps along the track → re-decode; converges in 1 iter). Wired into `rollout(..., scene_ctx=)`.
- Orientation SELECTION stays separate and open even at SOTA — the heightmap fixes CONTACT height, not facing.
  `probe_furniture_orientation.py` already pulls local scene geometry per clip; same plumbing.
- Orientation SELECTION stays separate and open even at SOTA — SceMoS leans on the planned
  approach for it, same as us. So the placement-side "perceive orientation → set the approach/
  placement yaw" remains the near-term answer; the heightmap tokenizer fixes CONTACT, not facing.

## Open limitation — sit orientation (the model ignores which way furniture faces)

The model sits without knowing the furniture's facing, so it can sit backwards. Diagnosed with
`scripts/chaining/diag_sit_facing.py`: it FOLLOWS THE APPROACH DIRECTION (|sit facing − GT| 31°)
and IGNORES a commanded sit facing (14° of 180° flip). Cause: no orientation signal anywhere —
occupancy is a footprint. **Do NOT "fix" the demo by approaching from the GT seated direction —
that is a hack (peeks at GT, doesn't touch the model, doesn't generalize), rejected on
2026-08-20.** Proper fix = an orientation-aware scene rep the model consumes (RGB render /
oriented-object map), or making it depend on an explicit orientation input. RESULTS §11.

**Orientation is a DATA limitation, not a tuning one (settled 2026-09-01).** HUMANISE bakes in
approach≈sit-facing (people walk in facing where they'll sit: |approach − GT sit facing| median
**5°**), so the motion model cannot learn facing as an independent signal — the heading input was
learned-to-be-ignored for exactly this reason (RESULTS §11). Therefore orientation must come from
the PLANNER (perceive the furniture facing → set the approach; the sit then follows it), NOT the
motion model. **Feasibility tested — perception works ~as well as it can, imperfectly:**
- Geometry heuristic (`probe_furniture_orientation.py`, "face away from the backrest mass"): median
  36°, 58% <45°, but **41% of furniture is ambiguous** (no backrest → returns None). Free, instant,
  deterministic, and it KNOWS when it's ambiguous.
- VLM (ollama `qwen3.5:27b`, vision, `think:False`; oblique render + projected world-compass,
  `tmp/vlm_orient_probe.py`): median **29°**, 58% <45° — **comparable to geometry, does NOT beat it**,
  slow (~13 s/query), and it always guesses (occasional 180° front/back flips, e.g. a toilet 150° off;
  scan quality hurts). Clear-fronted furniture (sofa/chair/bed/desk-chair) lands <30°; tables are
  unresolvable (no defined front).
- **Conclusion: "true sit" IS reachable via the planner on CURATED clear-fronted furniture** (not a
  hack, not motion-tuning): use the geometry heuristic (preferred — free, abstains on ambiguous) or
  the VLM to set the approach direction, restrict demo furniture to sofas/chairs/beds. ~30° median /
  ~40% notable-miss rate means it's demo-grade with curation, not production-robust. Top-down renders
  are illegible for facing; oblique perspective is needed (viewpoint selection required).

**BUILT & TESTED the orientation-driven demo (`scripts/chaining/demo_orient_sit.py`, 2026-09-01).**
The mechanic is PROVEN: **the sit obeys its planned facing to median 4°** (the model sits facing its
approach direction, so approaching IN direction F from the back side makes the sit face F — a new
`rollout(seg_headings=...)` hook forces a per-segment heading; approach-from-front + a forced sit
turn does NOT fire the sit). One clean end-to-end "true sit": scene0380 chair, perceived 5° from GT →
sat 2° from GT, fired (SAT&STOOD), no GT peek. **But it's narrow and low-yield, three compounding
caps:** (1) perception median 37° (great on chairs, bad on tables/couches); (2) sit FIRING only ~25%
(the walk→sit seam, RESULTS §11); (3) needs a walkable BACK side (freestanding furniture) so the sit
can approach facing F without a turn — ~⅔ of seeds skipped (ambiguous or wall-backed). Net: a correct
correct-facing fired sit lands on maybe ~1 in 8 seeds. **Conclusion: "true sit" is achievable but only
in a curated freestanding/clear-furniture slice at low yield — the user's "we can't truly have sit"
is largely right for the general case.** Harvestable for a demo clip or two, not robust.

## Heading (moonwalk) — FIXED at inference 2026-08-19 (RESULTS §11)

The body did not turn to face travel (|facing−travel| 78°, "moonwalking" on free chains).
- Target-heading CONDITIONING (`cond_mode=full_action_head`, model `step11/checkpoints/head_action`)
  was tried and FAILED: helped at 2k (56°) but the converged 20k model ignored it (76°) — the
  heading target is redundant with goal+prefix on GT, so it gets no gradient. Don't re-add it.
- The INFERENCE re-orient fixes it: `rollout(..., reorient=True)` (exposed as `--reorient` on
  `demo_interaction.py` and `diag_heading.py`) rotates each walk segment's start to face its
  goal. |facing−travel| 78°→**3°**, keeps clean seams (prefix is heading-canonicalized).
  Reoriented demo: `~/wander_data/step11_demo_reorient/`.
- `head_action` and the earlier `action` model behave the same on heading (conditioning ignored);
  use either with `--reorient`. Diagnose with `scripts/chaining/diag_heading.py [--reorient]`.

## The finding, in one paragraph (so a cold reader gets it)

Done-criteria 4/5 were blocked by TWO things, both invisible to goal error (which is (x,y)-only
and cannot see sitting, a z event — RESULTS §11): (1) goal augmentation (§10) trained pure
xy-reaching and suppressed sitting 75%→42%; (2) the walk→sit seam is out-of-distribution —
HUMANISE sit clips start from a standstill, so a mid-stride walking prefix drops sit 70%→0% and
the model just keeps walking. Fix (training-time only, no re-tokenize): `cond_mode=full_action`
(a 4-way action one-hot as an explicit "sit now" signal) + `--walk-prefix-aug` (swap a walking
prefix onto interaction clips to synthesize the missing seam) + `--goal-aug` restricted to walk
clips. Validated at 2k iters: walking-prefix sit 0%→85%.

## New/changed artifacts this cycle

- `scripts/chaining/demo_interaction.py` — composed walk→sit→stand→walk demo, pelvis-z SAT/STOOD
  structure check (no oracle exists for a composed chain, so structure is the check). Has
  `--start-dist` (synthesize a far chain start so the walk-up is real) and an AUTO-SPLIT walk-up
  into ≤1.1 m hops so the body is delivered to the furniture before the sit.
- `scripts/chaining/eval_sit_capability.py` — single-segment sit/stand by pelvis height, with
  `--prefix-mode {own,walk}` (the prefix-isolation control) and multi-ckpt compare.
- `scripts/chaining/eval_accumulation.py` — additive `--by-action` flag (default output unchanged);
  now loads BEV for `full_action`/`full_action_head` too.
- `scripts/chaining/diag_heading.py` — measures body-facing vs travel direction (the moonwalk
  diagnostic); `--reorient` to test the inference heading fix.
- `scripts/track1/add_goal_heading.py` — adds the target-heading field to a token manifest
  (geometry only, no re-tokenize); produced `~/wander_data/step11/tokens_head`.
- `scripts/track1/train_probe.py` — `cond_mode=full_action` and `full_action_head`,
  `--walk-prefix-aug`, goal-aug now walk-only. `scripts/chaining/rollout.py` —
  `rollout(..., actions=[...], reorient=...)`, `build_cond` action + heading args.
- Best models: `~/wander_data/step11/checkpoints/head_action` (latest; action+heading, use with
  `--reorient`) and `.../action` (action only) — the two behave the same on heading since the
  heading CONDITIONING is ignored at convergence (RESULTS §11). `step10/checkpoints/goalaug`
  remains best for pure navigation but sits only 42%.
  `step10/checkpoints/goalaug` remains best for pure navigation but sits only 42%.

## Collision-guided decoding — DONE (2026-09-02, RESULTS §13)

Implemented in `scripts/chaining/collision_guided.py` as SEGMENT-level rejection sampling
(guided_seg) plus the whole-chain floor (reject_chain), both vs greedy and the straight-line
oracle on identical scenes. guided_seg = per-segment best-of-N on `goal_err + w·collision`
(candidate 0 = greedy, so never worse than greedy). Built on `step10/checkpoints/goalaug`.
Result: below the straight-line oracle on both seeds while improving goal-following; see the
banner at the top and RESULTS §13 for the full table + caveats. `demo_rollout.py` still prints
the straight-line control for the greedy baseline number.

## Ready to use

- **Best model**: `~/wander_data/step10/checkpoints/goalaug` — full conditioning + goal
  augmentation. 0.374 m on arbitrary goals, 0.0560 m on familiar ones, 78 mm seams.
- **Collision map**: `~/wander_data/bev_tall_cache` (0.9 m threshold, 643 scenes).
  **Never score on the 0.12 m map** — it counts sitting on the target as a collision.
- **Chaining**: `scripts/chaining/{rollout,eval_accumulation,demo_rollout,render_chain_video}.py`
- **Data with goal augmentation support**: `~/wander_data/step10/tokens` (carries `xy_traj`)

## Not worth redoing

- **Target-instance exclusion for collision** — blocked, no ScanNet instance segmentation on
  either box. The connected-component proxy failed (331/400 clips merged furniture with walls);
  it survives as `src/target_occupancy.py` with the failure documented.
- **Scene ablation on single HUMANISE segments** — every metric is saturated or
  start-pose-determined there (RESULTS §8). Only chained rollouts resolve it.
- **Randomising the goal while keeping the motion** — teaches the model to ignore the goal.
  Truncation augmentation is the correct form (RESULTS §10).

## Where things live on the 3090

**2026-09-01: the data was MOVED to `~/Khiem/wander_data` (shell history: `mv wander_data Khiem`).**
Restored transparent access with a symlink `~/wander_data -> ~/Khiem/wander_data`, so every hardcoded
`~/wander_data/...` path (code, `.wander_env`, docs) still works. If a fresh box is missing `~/wander_data`,
recreate that symlink first. Nothing was lost.

All paths under `~/wander_data/` unless noted. None of it is in git.

| path | what |
|---|---|
| `motion_data/` | H3D, HUMANISE, scannet, `HUMANISE_263_cache` |
| `motion_data/track2_checkpoints/net_iter020000.pth` | **the finetuned VQ-VAE** — used by everything downstream |
| `track1_probe/tokens/` | tokens from the FROZEN tokenizer (RESULTS §4) |
| `track1_probe/tokens_finetuned/` | tokens from the finetuned tokenizer (RESULTS §5) |
| `track1_probe/checkpoints/` | `unconditioned`, `conditioned`, `conditioned-rel`, `unconditioned-ft`, `conditioned-rel-ft` |
| `continuation/tokens`, `continuation/checkpoints/` | RESULTS §7 — `noprefix`, `continuation` |
| `step8/tokens`, `step8/checkpoints/` | step 8 — `full` and `noscene`, both complete |
| `bev_cache/` | 643 scene renders (rgb + occupancy + extent), ~1.6 s/scene to regenerate |
| `deps/` | DINOv2 weights (ViT-S/14, ViT-B/14) + the patched repo — **moved out of /tmp** |
| `report_videos/`, `report_gallery/` | rendered outputs, synced to the Mac at `~/Documents/wander-output` |

### DINOv2 gotchas (env is Python 3.8 / torch 1.12)

`deps/dinov2_repo` is the upstream repo **patched** to run here — every file has
`from __future__ import annotations` prepended (upstream uses 3.10 `X | None` syntax).
It also needs a `scaled_dot_product_attention` shim (torch ≥2.0 API); that lives in
`scripts/scene_probe/scene_probe.py::_install_sdpa_shim`. Weights load `strict=True`,
which is the check that the architecture matches rather than silently falling back to
random init.

### Network

The per-flow throttle in CLAUDE.md §8 is real but is **not** a bandwidth cap. `aria2c -x8 -s8`
pulled 88 MB in <25 s and 346 MB in ~2 min. Use it for large single files.

---

## Known gaps, in priority order

1. **Scene conditioning is unevaluated and untestable on HUMANISE** — see RESULTS §8. Only
   0.8% of clips walk >1.5m; needs chained rollouts.
2. **Nothing has been chained.** Every result is single-segment or one-seam.
   Accumulation over N segments is the open research question (CLAUDE.md risk #1).
3. **No comparison to published work.** Generation FID unreproduced after 5 attempts;
   PSMo / AffordMotion untouched. Done-criterion #1 is half met.
4. **Single seed, single config everywhere.** Every probe is one run. Large effects
   (155.8→71.2, DINOv2≈raw pixels) would survive a reseed; smaller ones quoted in the docs
   (0.164→0.132, ratio 1.32→1.23) are point estimates with no run-to-run variance measured.
5. **No independent review.** The 90° SE(2) bug was found by auditing someone else's code.
   The code written since has been checked only by its own author, with oracle controls as
   the main defense.
