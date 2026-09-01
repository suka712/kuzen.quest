# CLAUDE.md — Scene-Aware Text-to-Motion Project

Authoritative project document. Read fully every session before acting. This is the
ground truth for goals, architecture, what is validated, and where the project is likely
to fail. Step specs are separate and disposable; this file is not.

**Status 2026-08-18.** Steps 1-11 done. All three original research bets passed — goal
grounding, conditional continuation, and chaining. **Step 11 (interaction in a chain) is DONE:
done-criteria 4 and 5 MET** — watchable walk→sit→stand→walk clips in real ScanNet rooms
(`~/wander_data/step11_demo_multiseg/`). It was NOT free engineering: interaction was blocked
by two obstacles both invisible to goal error (goal-aug suppressed sitting; the walk→sit seam
is OOD in HUMANISE), fixed by an explicit action one-hot + a synthesized walk→sit seam
(`cond_mode=full_action`, RESULTS §11). Best interaction model:
`~/wander_data/step11/checkpoints/action`.

**Step 12 (collision-guided decoding) is DONE (2026-09-02, RESULTS §13)** — the last SWAPPABLE
contribution. Inference-time steering (`collision_guided.py`, no training): greedy chains collide
≥ a straight line (the model doesn't steer), but per-segment best-of-N on `goal_err + 10·collision`
(`guided_seg`) drops collision below the straight-line oracle on both seeds while improving goal
error. **Next: step 13 (Qwen JSON → end-to-end ScanNet demo) and/or the paper writeup.** No number
here is comparable to published work yet (generation FID still unreproduced).

*(This header goes stale faster than anything else in the file. Three stale "next step"
pointers were found in one day. If it disagrees with `docs/IN_FLIGHT.md`, IN_FLIGHT wins —
and fix this line.)*

**Read `docs/IN_FLIGHT.md` FIRST** — running jobs, on-disk layout, next command.
`docs/RESULTS.md` has every established number with the oracle it is quoted against. This
file is authoritative for *what we are building*; RESULTS.md for *what is true*.

A note on epistemic status: several original assumptions turned out wrong (a frozen
tokenizer was assumed sufficient; cross-segment chaining was assumed trivial). Both were
wrong. This document reflects the corrected understanding. Treat the design as correct
but not sacred — components marked SWAPPABLE can change if evidence says so. Components
marked LOAD-BEARING cannot change without re-deciding the whole project. When a
LOAD-BEARING assumption looks false, STOP and surface it, do not work around it.

The inverse also applies, and has happened once: **when a result says a LOAD-BEARING
component fails, verify the measurement before believing it.** Track 1's "grounding is
impossible, re-architect" was an artifact of a 90° bug in its own eval. See section 5 #2.

---

## 1. Goal
Scene-aware text-to-motion for indoor scenes. **General full-body motion INCLUDING scene
interaction — sit, lie, reach-toward. Not navigation only.** A demo that only walks/turns
is a failure. Body-level interaction (22-joint skeleton) is in scope; dexterous hand/finger
manipulation is NOT in scope for V1.

To illustrate the expected input and output of this pipeline:
The pipeline is given a natural-language instruction, a 3D scene of the room, and the model's
starting position. An example input: **"I just woke up, I need to grab a cup of coffee and sit
down to start working."**

Expected output: the model stands up from the bed, walks to the kitchen area, then walks to the
desk and sits down. **The coffee itself is never picked up or carried** — "grab a cup of coffee"
motivates the kitchen waypoint only, not a depicted action. An actual grasp would be dexterous
hand/finger manipulation, already excluded from V1 above.

Deliverables:
- Explicit **start position** (x, y, yaw) as a direct user input.
- Explicit **goal** per motion segment (spatial coordinate, not inferred from text).
- **Indefinite chaining** of motion segments with continuous body pose across seams.
- **Local MLLM** planner (no cloud API).
- Open source.

---

## 2. The pipeline (read this before any code)

Two phases: offline TRAINING (produces the models), online INFERENCE (produces motion).
Understand both before touching either.

### 2a. Motion representation (LOAD-BEARING)
- Working skeleton is the **22-joint HumanML3D format**. SMPL-X inputs (HUMANISE) are
  reduced to these 22 body joints; hands and face are discarded. This is sufficient for
  sit/lie/reach at body level and matches every benchmark (HUMANISE, AffordMotion).
- Motion is encoded in the **263-dim HumanML3D feature vector**, which is CANONICALIZED:
  global translation and root orientation are removed; root motion is stored as local
  per-frame velocities. Consequence: the representation is position-invariant. A generated
  clip does not know where in the world it is. This single fact drives most of the design.
- Because of canonicalization, every clip is stored as TWO parallel tracks:
  1. Canonicalized 263-dim — the thing the model generates/reconstructs.
  2. World-frame root trajectory — per-frame absolute (x, y, yaw), Z-up ScanNet frame.
     Used for placement, chaining, goal/start conditioning, and all scene metrics.
- Yaw is always represented as `(sin, cos)`, never a raw scalar (avoids wraparound).
- **Collision is scored against the 0.9m TALL-obstacle raster**, never the default 0.12m one.
  At 0.12m the metric is void on an interaction dataset: ground-truth `lie` motion "collides"
  100% of the time because the person is on the bed, and being on the target IS the objective.
  At 0.9m walls survive and low furniture drops out. See 08 for the threshold sweep.
- **Two yaw conventions, 90° apart. LOAD-BEARING.** Canonicalized frame 0 faces +Z in
  HumanML3D's Y-up frame = **−Y in Z-up world = yaw −π/2**; but `compute_track2` defines
  **world yaw 0 = +X**. So placement rotates by **`yaw0 + π/2`** (`se2_utils.se2_place`,
  inverse `world_to_local_xy`). Getting it wrong rotates every trajectory 90° and is
  invisible to a start-error check — it already invalidated one track's conclusion.
- **Corollary, and the main design lesson so far: give the model geometry in the frame it
  generates in** — start-relative, heading-aligned. Absolute world coords force it to
  compute `R(yaw₀)ᵀ(goal − start)`, a bilinear op, and it will simply fail to. Measured:
  0.515 m absolute vs 0.164 m relative, everything else identical.

### 2b. Training phase (two stages, STRICT ORDER)

Stage A — **VQ-VAE (motion tokenizer) joint finetune.**
- What it is: the VQ-VAE encodes a motion clip into a sequence of discrete tokens (indices
  into a learned codebook) and decodes tokens back to motion. It is the vocabulary of
  motion the rest of the system speaks in.
- Why finetune (not freeze): the off-the-shelf VQ-VAE reconstructs HUMANISE interaction far
  worse than locomotion. **DONE and it worked** — lie 136.9 -> 96.3mm, every category
  improved, H3D held-out flat. A reportable result, not just a gate. See RESULTS §3.

Stage B — **Transformer finetune (conditional continuation).**
- What it is: the autoregressive model (the "GPT" of T2M-GPT) that predicts a sequence of
  motion tokens. In vanilla T2M-GPT it is conditioned on TEXT ONLY.
- What changes here — two additions, both LOAD-BEARING:
  1. **Explicit spatial conditioning.** The transformer receives, as learned embeddings
     concatenated to its input, the **goal coordinate expressed in the segment's own start
     frame** (start-relative, heading-aligned). The model is TOLD where to end. It does NOT
     infer location from text or from the scene image. (Decision locked: goal is an
     explicit input, never text-only. Inference burden is the enemy.)
     **VALIDATED — see 2f.** Note what is NOT fed: the absolute start pose. In the start
     frame it is (0,0) at heading 0 by construction, so it carries no information; it
     enters at inference only, as the SE(2) placement. Feeding absolute world coordinates
     instead does not work — measured, see 2a and RESULTS §4.
  2. **Conditional continuation for chaining.** The transformer is trained to generate a
     segment CONDITIONED ON THE TAIL of the previous segment (prefix = last N tokens / last
     pose of segment k-1). This is what makes chaining actually work — see 2d. This is a
     training-time change, not an inference trick; the model must LEARN to continue from an
     arbitrary ending pose.
  3. **Explicit action conditioning (added step 11, LOAD-BEARING for interaction).** A 4-way
     action one-hot (walk/sit/stand up/lie) concatenated to the cond vector (`cond_mode=
     full_action`). WHY it is not optional: the goal is only (x, y), so nothing else tells the
     model to SIT rather than walk to the seated xy — and at a walk→sit seam the walking prefix
     otherwise wins and it keeps walking (measured 85%→0% sit without it). It is the exact
     quantity the MLLM already emits (`action`), so it costs nothing at inference. Paired with
     `--walk-prefix-aug`, which synthesizes the walk→sit seam HUMANISE lacks. RESULTS §11.
- Scene conditioning also enters here, as an occupancy crop in the segment's own start
  frame (NOT DINOv2 features — see section 4 and RESULTS §6).
- STRICT ORDERING: finetuning the VQ-VAE changes the codebook, which invalidates all
  previously extracted tokens. After Stage A you MUST re-extract tokens with the finetuned
  VQ-VAE, THEN train the transformer on those new tokens. Training the transformer on stale
  tokens produces silent garbage. Order is always: finetune VQ-VAE -> re-extract tokens ->
  finetune transformer.

### 2c. Inference phase
1. **Qwen3-VL (local MLLM)** reads the instruction + a scene image and emits a per-segment
   JSON plan: `{action, goal_object_id, goal_coord: [x,y], duration}`. The MLLM does the
   spatial grounding — it decides WHERE the goal is. The motion model never guesses location.
2. For each segment, the **transformer** generates motion tokens, conditioned on: text
   + **goal coord expressed in that segment's start frame** (see 2b.1 and 2f) + (scene
   features) + prefix from previous segment. The start pose is not a model input; it is
   applied in step 5.
3. **Collision-guided decoding** (optional steering, see 2e) re-ranks candidate tokens
   against the scene occupancy map during generation.
4. **VQ-VAE decoder** turns tokens into canonicalized local motion.
5. **SE(2) rollout** places that local motion into the world by composing it onto the
   segment's start pose, producing world-frame motion.
6. **Chaining**: the end pose of segment k becomes the start pose (and prefix context) of
   segment k+1. Repeat for indefinite length.
7. Output: world-frame motion rendered to mp4 (or driving a robot).

### 2d. Chaining — why it is hard, and what actually works (LOAD-BEARING)
The naive approach (generate each segment independently, then just place segment k+1 where
segment k ended) DOES NOT WORK. Because motion is canonicalized, each segment is generated
as if starting from a neutral pose. Segment k ends mid-stride in a specific body
configuration; segment k+1 begins from canonical neutral. Gluing them makes the body
teleport at the seam — feet slide, limbs snap. Placing the root correctly with SE(2) fixes
position but not body configuration.

The working approach: the transformer generates segment k+1 CONDITIONED ON THE TAIL of
segment k. **VALIDATED — RESULTS §7.** Seam error 155.8 -> 71.2mm,
landing within 0.3mm of the oracle, so the conditioning extracts everything available.

Three things the probe settled, all load-bearing for step 8:
1. **Condition on the ending BODY CONFIGURATION, not on tokens.** Segments are
   canonicalized individually, so segment k's tokens describe a different frame and are
   ill-posed as a prefix. Root-relative, heading-canonicalized joint positions (66-d) are
   frame-independent. Express them in segment k+1's frame, which is computable at inference
   because k+1's canonical frame is DEFINED by the seam pose.
2. **The prefix pose must be ON-MANIFOLD.** A VQ-VAE-reconstructed prefix (structured
   ~70mm error, what real chaining supplies) costs only 15mm of seam. iid Gaussian noise at
   25mm — a fifth the magnitude, but anatomically impossible — collapses it to 331mm, flat
   out to 200mm, i.e. off-manifold rather than degraded. So never feed a blended,
   interpolated or smoothed pose forward as conditioning; feed the decoded one.
3. **The seam blend is NOT cosmetic.** ~70mm of the residual is VQ-VAE reconstruction of
   the first frame (oracle 70.9mm vs a 21.6mm canonicalization floor) and no amount of
   better conditioning touches it. The blend is what hides it.

### 2e. Collision-guided decoding — what it is and its status (SWAPPABLE, NOT the hill)
At inference, in the AR loop: instead of greedily taking the top next token, take the
top-k, decode each candidate's resulting root movement, check it against the scene
occupancy map, and re-rank to prefer non-colliding, goal-approaching motion. It injects
scene-awareness at decode time without extra training.

It is the most academically novel piece AND the least proven. It can fail two ways:
(1) if the top-k candidates are all similar (e.g. all "walk forward"), there is no evasive
option to promote; (2) greedy per-token re-ranking cannot plan a turn several tokens ahead.
Fallbacks, in order: beam search over a short horizon; segment-level rejection sampling
(generate N candidate segments, keep the lowest-collision one) — rejection sampling is the
guaranteed floor and always improves non-collision.

STRATEGIC NOTE: this is NOT the load-bearing contribution. If it underwhelms, it demotes
to an ablation ("+X% non-collision") and the paper still stands on explicit control +
conditional-continuation chaining + an interaction-capable tokenizer. Do not let the
project's success depend on this component.

### 2f. Grounding — VALIDATED (2026-08-12)
Explicit goal conditioning works: 0.164 m goal error against a 0.124 m oracle floor and a
0.490 m unconditioned baseline, holding across displacement scale. Retires the goal-grounding
risk. Says nothing about chaining. Numbers and caveats: `docs/RESULTS.md` §4-5.

---

## 3. What is validated

**All of it is in `docs/RESULTS.md`** — per-stage numbers, the oracle each is quoted against,
the five-entry bug ledger, and the gotchas that will bite again (VQ-VAE normalization, the
leaked 45.3mm baseline, `QuantizeEMAReset` not surviving `load_state_dict`, the 0.9m collision
threshold). Do not re-derive or re-litigate any of it here.

Obstacle avoidance is now DONE at DECODE time (step 12, RESULTS §13): `guided_seg` steering beats
the straight-line oracle on both seeds. Note the distinction — the trained scene CONDITIONING still
does not by itself steer (greedy collides ≥ a straight line); the steering is an inference-time
selector on top. Still not established, and worth stating plainly: **the composed-chain interaction
yield is ~50% and single-seed** (RESULTS §11), **step 12 is two seeds / one model**, and
**generation FID is still unreproduced after 5 attempts**, so no number in this repo can be
compared to published work. (Chaining, interaction-in-a-chain, and steering ARE done — §9, §11, §13.)

---

## 4. Components and how locked they are
LOAD-BEARING (changing these = re-deciding the project):
- 22-joint HumanML3D representation with two-track (canonical + world-frame) storage.
- VQ-VAE finetuned (not frozen), joint on HumanML3D + HUMANISE.
- Transformer with explicit spatial conditioning (goal in the start frame — see 2a, 2f),
  conditional continuation for chaining, AND explicit action conditioning (the action one-hot,
  step 11 — see 2b.3). Interaction in a chain does not happen without the action input.
- Strict training order: VQ-VAE -> re-extract tokens -> transformer.

SWAPPABLE (change freely if evidence says so — surface it, don't agonize):
- ~~**Scene representation.**~~ **DECIDED 2026-08-12, was swappable, now settled** —
  RESULTS §6. The inherited BEV-RGB + DINOv2 encoder was probed and does not
  carry usable scene signal: it fails to beat its own unencoded input (37.6% / 34.3% vs
  36.4% raw pixels, 26.4% prior), and ViT-B scores *below* ViT-S, so it is the domain, not
  capacity. The **binary occupancy raster, cropped in the agent's own frame**, reaches 63.0%
  on the same probe. Scene conditioning now uses occupancy. This also removes a ViT from the
  inference loop and reuses the raster that collision-guided decoding already needs. The RGB
  render stays available for figures, not for conditioning.
- Collision-guided decoding (see 2e) and its fallbacks.
- MLLM choice (Qwen3-VL 8B) — swappable if planning quality is poor.
- Seam-blend details.

NOT excluded by default: earlier drafts listed RRT*, retrieval DB, and heightmap as
"excluded." That was leftover cruft from a diluted context, not a real decision. Nothing
is excluded on principle. Add what helps; justify it.

---

## 5. Critical failure points — know these, back off early
Ordered by risk. For each, the plan is to test cheaply and pivot early, not to discover
failure after building everything on top.

1. **Chaining continuity (2d).** Mechanism VALIDATED at one seam
   (RESULTS §7) — but downgraded, not retired. What remains untested
   is **error accumulation over a real chain**: the probe used a ground-truth previous
   segment, and its reconstructed-prefix proxy (86mm seam, 2x goal error) is one seam's
   worth of degradation, not N. Drift over an indefinite chain is now the open question.
4. **Measurement validity.** Five silent convention bugs so far (ledger in RESULTS.md), two
   caught only after they had changed a conclusion. Each was missed because the check used
   could not detect it — start-error cannot see a rotation; per-frame MPJPE cannot see
   cumulative drift. **MANDATORY: every pipeline that emits a number gets an oracle
   control** — push ground truth through the identical path and confirm near-perfect
   output BEFORE reading any model number off it. If the oracle is not small relative to
   the effect you are measuring, you have no measurement. Highest-frequency failure mode
   in this project, ahead of any algorithmic risk.
5. ~~Collision-guided decoding (2e). May not steer.~~ RETIRED 2026-09-02 — it steers (RESULTS
   §13). Both the guaranteed floor (rejection sampling) and the reactive `guided_seg` selector
   beat the straight-line oracle; `guided_seg` also improves goal error. Did NOT demote to an
   ablation; it is a working contribution. The design worry was right that greedy alone does not
   steer — that is exactly the baseline it beats.
6. **Shared GPU.** The 4090 is shared; the 3090 is not (but as of 2026-09-02 another user's
   light job runs on the 3090 too — check `nvidia-smi --query-compute-apps` first). See section 8.
7. ~~Goal grounding~~ RETIRED 2026-08-12 — probe passed, see 2f. Do not pivot to
   trajectory-first.
8. ~~VQ-VAE joint finetune balance~~ RETIRED 2026-08-12 — Track 2 passed at 1:1 sampling,
   lr 2e-5, no forgetting.

Meta: chaining, goal-following, interaction-in-a-chain (step 11, RESULTS §11), AND obstacle
avoidance (step 12, RESULTS §13) are all retired. **The core research bets are all closed.**
Obstacle avoidance was the last: `guided_seg` decode-time steering drops collision below the
straight-line oracle while improving goal error (§13). What remains is integration (step 13,
Qwen→demo), benchmark comparison (step 14), and the writeup — engineering plus discipline about
measurement (#4), which in step 11 caught the "sit" that was really a walk (goal error is z-blind;
use pelvis height) and in step 12 kept goal_err in the steering objective so it can't "avoid" a
wall by refusing to move.

---

## 6. Build order

**Done (1-11).** Numbers in `docs/RESULTS.md` §1-11: plumbing · baseline calibration
(reconstruction only) · tokenizer finetune · grounding probe · token re-extraction · scene
probe · continuation probe · combined transformer · chaining · goal augmentation ·
interaction-in-a-chain.

Steps 4/6/7 were **probes** — one conditioning input each, small budget, single seed. Step 8
was the first model trained on all of it; step 10 fixed arbitrary-goal navigation; step 11
(`~/wander_data/step11/checkpoints/action`, `cond_mode=full_action`) is the current best and
the first that INTERACTS in a chain.

11. ~~**INTERACTION IN A CHAIN.**~~ DONE (RESULTS §11, done-criteria 4/5 met). Turned out to be
    real research, not a demo script: goal augmentation had suppressed sitting and the walk→sit
    seam was OOD (both hidden by goal error, which is z-blind). Fixed with an explicit action
    one-hot + a synthesized walk→sit seam (`--walk-prefix-aug`) + walk-only goal-aug.

12. ~~**Collision-guided decoding**~~ DONE (RESULTS §13, 2026-09-02). Inference-time steering,
    no training (`scripts/chaining/collision_guided.py`). `guided_seg` = per-segment best-of-N on
    `goal_err + 10·collision` (greedy is always candidate 0). Drops collision from greedy's
    2.06%/3.63% to 0.69%/2.64% — BELOW the straight-line oracle (1.57%/3.93%) on both seeds — while
    IMPROVING goal error to 0.09–0.11 m. `reject_chain` (whole-chain rejection) is the floor but
    degrades goal error. Figure: `~/wander_data/step12_fig/cg_compare_scene0001_00.png`.

13. ~~**Qwen JSON** wired end-to-end~~ DONE (RESULTS §14, 2026-09-02). `scripts/planner/` — an
    instruction + the scene image go to a LOCAL VLM (ollama `qwen3.5:27b`), which grounds the goal
    by picking a numbered furniture ANCHOR (low-furniture connected components, `scene_anchors.py`)
    and emits a `{action, target}` segment plan; `demo_end2end.py` expands it, runs the step-11
    action model with `guided_seg` steering on walks, and renders in the room mesh. Verified: qwen
    grounded "the couch" correctly; sit/stand fired by pelvis height (SAT 0.56 m, STOOD 0.95 m).

**-> YOU ARE HERE. Next: 14 (benchmark + FID), or polish (skinned body mesh, sit-placement).**

14. **Benchmark comparison** (PSMo / AffordMotion) + generation FID. **Check how they define
    non-collision before quoting anything** — given RESULTS §8 theirs cannot be the naive
    definition, and the definition decides comparability.

## 7. Done criteria
1. ~~Baseline matches T2M-GPT paper~~ DONE for reconstruction; harness trusted.
2. ~~VQ-VAE reconstructs interaction (sit/lie) acceptably after joint finetune~~ DONE.
3. ~~Single segment reaches an explicit goal from an explicit start pose~~ DONE at probe
   scale with the frozen tokenizer (2f). Re-confirm on the finetuned tokenizer after
   token re-extraction.
4. Multi-segment instruction -> correct per-segment motion **including interaction**, with
   continuous body pose across seams. **MET** (RESULTS §11). `demo_interaction.py` chains
   walk→sit→stand→walk with explicit per-segment actions; 5/10 chains sit AND stand, verified
   by pelvis height (goal error is z-blind to sitting — that blindness is what hid the whole
   problem for a day). Interaction seams are larger than walk seams (92–155 mm vs 60) and the
   display blend hides them.
5. Watchable mp4 of **scene interaction** in a ScanNet room. **MET** —
   `~/wander_data/step11_demo_multiseg/` (`demo_00` scene0151/couch, `demo_05`
   scene0694/coffee-table, both 0% collision).

   It did NOT "work as-is" (an earlier draft of this line guessed it might). Two obstacles had
   to be fixed first — goal-aug suppressing sitting and the OOD walk→sit seam — via an explicit
   action input and `--walk-prefix-aug`. The lesson stands: sample goals AT furniture, give each
   segment its action EXPLICITLY (not just via text — the action one-hot is load-bearing), and
   deliver the body to the furniture with in-range walk hops before asking it to sit.

## 8. Environment / logistics
- Data: `/media/user/2tb/motion_data/` (HUMANISE, HumanML3D, ScanNet meshes). Data,
  checkpoints, renders live on disk and are gitignored — never committed.
- T2M-GPT base code: `/home/user/Khiem-ssh/T2M-GPT/` — kept SEPARATE from the working
  `wander` repo; wander imports/points at it, does not fork it in.
- Hardware: two boxes on Tailscale, both ssh-able from the Mac as `train-4090` (`ntx`) and
  `train-3090` (`dsx`).
  - **4090**, 25 GB, ran Track 2. **Shared with another user's job** — check
    `nvidia-smi --query-compute-apps` before blaming your code for a stall. Light jobs get
    cycles under contention; autoregressive generation may not.
    Paths: `/media/user/2tb/motion_data`, `/home/user/Khiem/T2M-GPT`.
  - **3090**, 24 GB, not shared, ran Track 1. Env vars in `~/.wander_env` (source
    explicitly — non-interactive shells miss it). The `afford` conda env was raw-copied
    from the 4090, so bare `pip` is broken; use `~/anaconda3/envs/afford/bin/python -m pip`.
    Paths: `~/wander_data/motion_data`, `~/Khiem/T2M-GPT`, repo `~/Khiem/wanderscript-text-to-motion`.
  - **Network**: both boxes are behind an upstream **per-TCP-flow** rate limit, so a
    single-connection `git clone`/`curl` crawls. It is NOT a bandwidth cap: `aria2c -x8 -s8`
    pulled 88 MB in <25 s and 346 MB in ~2 min from a Range-capable host. Use aria2c for
    large single files, rsync from the other box over the Tailscale direct path for
    directories, and expect plain `git clone` of a big repo to be slow. This repo is tiny,
    so `git pull`/`push` is unaffected.
- Benchmarks: PSMo + AffordMotion (reported HUMANISE numbers; PSMo has no public code, so
  state test-protocol differences honestly). SceMoS (CVPR 2026, arXiv 2602.20476) is related
  work only — it reports on TRUMANS, a different dataset; no numeric comparison. **But its
  architecture is the pointed lesson for our scene-awareness gap:** it grounds the *tokenizer*
  in a local scene heightmap (VQ-VAE decoder takes `(token, heightmap)`; ±0.6 m body-frame,
  32×32, contact-indicator loss), so decoded motion is contact-correct. Ours puts scene only on
  the *transformer* as an occupancy footprint — height/orientation blind — which is why the
  model is scene-grounded but not contact/geometry-aware. See docs/IN_FLIGHT.md "Next direction".
