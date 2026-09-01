# Results

---

# PART 1 — Plain English

## What we're building

You type an instruction. A person moves realistically inside a real scanned 3D room —
walking, and also sitting and lying on the actual furniture.

Three pieces:
1. A **tokenizer** turns motion into a small alphabet of "motion words".
2. A **transformer** writes sentences in that alphabet — it generates the motion.
3. A **planner** (later) reads your instruction and decides where in the room to go.

Motion is generated in short segments and glued together, so the person can be told to do
several things in a row.

## What works right now

- **The tokenizer handles furniture interaction far better after retraining.** Reconstructing
  someone lying down improved ~30%, and nothing else got worse.
- **Segments join smoothly.** Gluing two clips naively makes the body snap and teleport. Our
  fix — telling the next segment what body pose to start from — cuts that error in half, and
  it's as good as the tokenizer physically allows.
- **Long chains don't fall apart.** Chaining 10 segments gives a 7–9 m walk through a real
  room, and error does *not* build up as the chain gets longer.
- **We dropped a component that wasn't earning its place.** The inherited scene encoder
  (DINOv2) turned out no better than feeding raw pixels. Plain geometry — a floor plan of
  what's solid — works nearly twice as well and is free.
- **It now walks to targets it was never trained on.** This was broken as of yesterday — the
  model moved but not toward the goal. Training it on randomly-cut clips fixed it: error on
  unfamiliar targets more than halved, and it now walks the distance it is told to.

## What doesn't work

- **It doesn't avoid furniture.** It bumps into things slightly *more* often than if it just
  walked in a straight line. Nothing in the system currently steers around obstacles — that is
  the next piece of work.
- **Nothing is comparable to other people's published results yet.** One benchmark number has
  resisted five attempts to reproduce. Every number here is internally consistent but can't
  yet be put next to another paper's.

## What's next

Teach it to steer around furniture. Everything else is in place: it goes where it is told,
segments join cleanly, and chains hold together. Bumping into things is the last gap before a
demo worth showing.

## Reading the numbers below

Six terms do most of the work in Part 2:

| term | meaning |
|---|---|
| **oracle** | The best score physically possible — computed by feeding the *right answer* through the same machinery. If our model matches the oracle, the remaining error isn't the model's fault. |
| **null** | The score for doing nothing (e.g. never moving). If a model can't beat this, it isn't doing anything useful. |
| **in-distribution** | Similar to what the model was trained on. **Out-of-distribution** means unfamiliar — models often look great on the first and fail on the second. |
| **saturated** | Our model is already as good as the oracle, so the test can no longer tell two models apart. A "no difference" result then means the test is blind, not that the models are equal. |
| **seam** | The join between two motion segments. "Seam error" = how much the body jumps there. |
| **drift** | How far the person ends up from where they should have. |

Distances are metres (m) or millimetres (mm). Lower is better everywhere.

---

# PART 2 — Technical detail

One section per build-order stage. Result first, then only the method details that change how
a number should be read. Reproduce commands live in the script docstrings, not here.
Superseded per-stage docs are in `archive/`.

Every number is held-out. Every generative number is quoted against an **oracle** — a model
number without its oracle is not interpretable, and this project has twice published one that
turned out to be measuring a bug.

---

## 1 · Data pipeline — DONE

| check | result |
|---|---|
| HUMANISE 3-source ID join | 19,648 / 19,648 |
| 22-joint → 263-dim vs HumanML3D's own vectors | **0.80 mm** MPJPE, foot contact bit-exact |
| HUMANISE → 263 at scale | 19,648 / 19,648, 0 NaN |
| world-frame track vs scene-mesh floors | 150 / 150 scenes |
| ScanNet meshes load / BEV world→pixel error | 643 / 643 · 0.67 px |

Two-track storage (CLAUDE.md 2a): canonicalized 263-dim for the tokenizer, plus a world-frame
`(x, y, yaw)` root track, because the 263-dim is position-invariant and cannot know where it
is. The converter wraps HumanML3D's own extractor rather than reimplementing it.

**Caveat:** the upstream SMPL-X → 22-joint reduction is HUMANISE's own and has never been
independently verified. It underlies every HUMANISE number here. No longer the prime suspect
for anything (see §3), but unchecked. Also: 2 NaN-corrupted files exist in HumanML3D.

## 2 · Baseline calibration — DONE (reconstruction only)

Reconstruction FID **0.066** vs paper 0.070; R@1–3 within variance. Harness trusted.

Frozen tokenizer, held-out MPJPE: H3D 56.11 · H3D-lie 117.46 (n=11) · HUMANISE walk 50.20 ·
stand-up 66.98 · sit 69.55 · **lie 136.91**.

**Two traps, both of which have already cost this project real time:**
- **Normalization.** The VQ-VAE expects its *own* checkpoint mean/std
  (`checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/`), never H3D's `Mean.npy`/`Std.npy`.
  Wrong pair inflates everything ~2-3× (H3D reads 137 mm instead of 45). Suspect this first
  when a reconstruction number looks absurd.
- **The old 45.3 mm H3D baseline was LEAKED** — sampled from all of `new_joint_vecs`,
  including training clips. Honest held-out figure is **56.11**; H3D-lie is **117.46**, not
  90.1. Do not quote 45.3 or 90.1.

**Still open:** generation FID / R-precision (autoregressive, not encode/decode) — unreproduced
after 5 attempts, root-caused to a self-inflicted timeout. Nothing downstream needed it, but it
means **no number here is comparable to published work.**

## 3 · Tokenizer joint finetune — DONE, works

HumanML3D + HUMANISE, 1:1 balanced, lr 2e-5, 20k iters. Held-out:

| | H3D | H3D-lie | walk | stand up | sit | lie |
|---|---|---|---|---|---|---|
| frozen | 56.11 | 117.46 | 50.20 | 66.98 | 69.55 | 136.91 |
| finetuned | 56.2 | 117.4 | **34.1** | **53.0** | **47.6** | **96.3** |

Both gates met: interaction improved, nothing regressed. Lie holds a 92–101 mm band across the
last 9 of 11 checkpoints. **The sit/lie gap was real codebook coverage, not the unverified
upstream joint reduction** — finetuning the codebook alone closed most of it.

**Landmine:** `QuantizeEMAReset` keeps `init`/`code_sum`/`code_count` as plain Python
attributes, not buffers. `load_state_dict(strict=True)` restores the codebook but leaves
`init=False`, so the first `train()` forward **overwrites all 512 codes from one batch**.
Anyone resuming or finetuning this VQ-VAE must seed the EMA accumulators from the loaded
codebook first (`prepare_quantizer_for_finetune`).

**Caveat:** the 64-frame window keeps only 38.3% of HUMANISE train clips (lie 49%, sit 50%,
stand-up 23%, walk 33%). Sit/lie survive better than walk, so it does not bias against the
categories of interest.

## 4 · Grounding — DONE, works

Frozen tokenizer, single segment, 200 held-out clips:

| | goal err | corr(commanded, achieved) |
|---|---|---|
| NULL (stay at start) | 0.627 m | — |
| unconditioned | 0.490 m | +0.07, +0.62 |
| conditioned, **absolute** frame | 0.515 m | −0.12, +0.60 |
| conditioned, **relative** frame | **0.164 m** | **+0.90, +0.97** |
| ORACLE | 0.124 m | — |

**The one decision that matters: feed the goal in the frame the model generates in** —
start-relative, heading-aligned (`se2_utils.world_to_local_xy`). Absolute world coordinates
force it to compute `R(yaw₀)ᵀ(goal − start)`, a bilinear op, through a single `Linear` that
cannot represent a rotation; it does not learn this. The start pose is **not** a model input —
in that frame it is (0,0) at heading 0 by construction, and enters only at SE(2) placement.

Holds across range (600 clips): 0.071 m error at <0.25 m commanded → 0.508 m at >2 m, with the
model/oracle ratio flat at 1.1–1.4. **Goal accuracy is limited by tokenizer reconstruction,
not by goal-following.**

**The 90° bug, kept because it explains the discipline.** The first pass reported grounding
FAILS and recommended re-architecting the project. Its `se2_place` rotated by `yaw0` instead of
`yaw0 + π/2`, rotating every trajectory 90°. Start-error is exactly 0.0 under *any* rotation,
so the only sanity check run was structurally blind to it. A ground-truth-token oracle catches
it instantly: 0.862 m at offset 0 vs 0.124 m at +π/2 — i.e. the bug alone accounted for the
entire "model" error of 0.828 m.

## 5 · Token re-extraction — DONE, verified

Stage A changed the codebook, so all prior tokens are invalid. 19,648 clips re-tokenized;
frozen tokens kept separately so §4 stays reproducible.

Verification (file sizes prove nothing — token count depends only on clip length): 512/512
codebook rows changed · only 80/3125 test clips have identical token sequences · 45.3%
per-token agreement · goals/starts unchanged.

Oracle floor **0.124 → 0.107 m**. Probe re-run on the new tokenizer: unconditioned 0.549 m,
**conditioned 0.132 m** — conditioning now cuts error 76% (was 67%), model/oracle ratio
tightening 1.32 → 1.23. **Stage A improved goal-reaching with no change to the grounding
mechanism.**

**The hazard this step prevents, quantified:** new tokens through the *old* decoder gives
0.321 m, 3× worse, with nothing raised. Stale tokens decode to something plausible rather than
crashing. Always name the tokenizer explicitly.

## 6 · Scene representation — DONE, DINOv2 dropped

4-way action classification from scene context at the goal, no text, **split by scene**, linear
head, 175 held-out scenes:

| arm | acc |
|---|---|
| prior | 26.4% |
| DINOv2 ViT-B/14 | 34.3% |
| RGB crop, **no encoder** | 36.4% |
| DINOv2 ViT-S/14 | 37.6% |
| **occupancy crop** | **63.0%** |

**DINOv2 does not beat its own unencoded input, and ViT-B scores below ViT-S** — a domain
problem, not capacity. Occupancy nearly doubles it and we already render it. Scene encoder
swapped to the **occupancy raster cropped in the agent's frame**: removes a ViT from the loop,
drops a 350 MB dependency, and reuses the raster collision needs anyway.

63% is near the ceiling for static geometry: sit and stand-up happen at the same furniture and
differ only in motion direction, so nothing static separates them (stand-up recall 0.51).

## 7 · Conditional continuation — DONE, works

The chaining mechanism, tested at one seam. 300 held-out clips:

| | seam err | goal err |
|---|---|---|
| no-prefix (naive) | 155.8 mm | 0.0707 m |
| **continuation, exact prefix** | **71.2 mm** | 0.0573 m |
| continuation, **reconstructed** prefix (realistic) | 86.0 mm | 0.1073 m |
| ORACLE | 70.9 mm | 0.0544 m |
| canonicalization floor | 21.6 mm | — |

Lands within 0.3 mm of the oracle — the conditioning extracts everything available.

**Three findings that constrain chaining:**
1. **Condition on the ending body configuration, not tokens.** Segments are canonicalized
   individually, so segment *k*'s tokens describe a different frame. Root-relative,
   heading-canonicalized joint positions (66-d) are frame-independent. Express them in *k+1*'s
   frame — computable at inference, since *k+1*'s canonical frame is *defined by* the seam pose.
2. **The prefix pose must be ON-MANIFOLD.** A VQ-VAE-reconstructed prefix (structured ~70 mm
   error) costs 15 mm. iid Gaussian noise at 25 mm — a fifth the magnitude but anatomically
   impossible — collapses it to 331 mm, *flat* out to 200 mm. So **never feed a blended,
   interpolated or smoothed pose forward**; feed the decoded one.
3. **The seam blend is not cosmetic.** ~70 mm of residual is VQ-VAE reconstruction of the first
   frame (oracle 70.9 vs a 21.6 mm floor) and no conditioning touches it.

**Canonicalization residual (21.6 mm):** `process_file` normalizes per segment —
`uniform_skeleton` rescales from each segment's first frame, floor offset is each segment's
minimum height — so two segments never agree exactly on the same physical pose. The floor term
is chicken-and-egg at inference. Seams below ~20 mm need the representation changed.

## 8 · Transformer finetune — DONE; scene arm UNTESTABLE on this data

All conditioning together (text 512 + goal 2 + seam pose 66 + occupancy 784 = 1364-d), 20k iters:

| | seam | goal |
|---|---|---|
| full (with scene) | 71.7 mm | 0.0584 m |
| noscene (ablation) | 71.4 mm | 0.0564 m |
| ORACLE | 70.9 mm | 0.0544 m |

**The combination does not degrade** — matches the 4k-iter §7 probe. A prior recorded before
the run predicted it would underperform; that prior was wrong.

**Ignore token accuracy (99.5%).** The seam pose nearly determines the early target. Never
quote it as quality.

**Why the ablation answers nothing — it is the data.** Both models sit *at* the oracle: 0.8 mm
and 4 mm of headroom. Goal error is also the wrong question for a scene arm, which is about
obstacles, not coordinates. Building a collision metric took three attempts:

1. *Exclude the target ScanNet instance* — **blocked**, no instance segmentation on either box.
2. *Connected-component proxy* — **failed**, 331/400 clips merged furniture with walls.
   Abandoned rather than tuned, since loosening the guard excludes the walls the metric exists
   to detect. Kept as `src/target_occupancy.py` with the failure documented.
3. *Tall-obstacle map* — **works.** Ground-truth collision by obstacle-height threshold:

| threshold | lie | sit | walk | raster occupied |
|---|---|---|---|---|
| 0.12 m (default) | 100.0% | 88.6% | 30.2% | 22.7% |
| **0.90 m** | **20.1%** | **3.9%** | **0.1%** | **6.1%** |
| 1.50 m | 0.0% | 2.6% | 0.0% | 1.5% |

**Collision must be scored at 0.9 m**, never 0.12 m. At 0.12 m the metric is void on an
interaction dataset — ground-truth `lie` "collides" 100% because the person is on the bed, and
being on the target *is the objective*. Cached at `~/wander_data/bev_tall_cache`.

With a correct metric the ablation *still* returns nothing: full 3.09% · noscene 3.09% ·
ORACLE 3.09% · **NULL (never move) 3.25%**. NULL landing in the same place is the tell —
collision is a property of the **start pose**, not the path. **Only 25 of 2962 clips (0.8%)
walk further than 1.5 m**; mean displacement is 0.63 m.

**This one fact explains every failed step-8 evaluation.** Goals are 0.6 m away so goal error
saturates; nothing moves so collision is start-determined; there is no navigating to do so the
scene arm has nothing to do. **It is the absence of a test, not evidence that occupancy
conditioning fails.**

Consequences: HUMANISE alone cannot demonstrate scene-aware navigation. **Chaining is the
unlock** — composing segments is what produces multi-metre paths and therefore the first
setting where a scene arm is measurable. And **check how PSMo/AffordMotion define non-collision
before any comparison**; theirs cannot be the naive definition.

## 9 · Chaining — DONE. Works, and does not accumulate

The first motion in this project longer than one segment. Each segment is conditioned on the
previous segment's **decoded** ending pose; heading is recovered from the generated body via
the same hip/shoulder formula as `compute_track2`, so the chain's own geometry decides where
it points.

### Accumulation — the open research question, answered

Real clips cut into N consecutive segments, each fed its true endpoint as goal, chained from
the first segment's real start:

| N | final drift | ORACLE drift | NULL | seam | goal | n |
|---|---|---|---|---|---|---|
| 1 | 0.281 m | 0.113 m | 0.659 m | — | 0.281 m | 200 |
| 2 | 0.252 m | 0.154 m | 0.701 m | 71.9 mm | 0.215 m | 168 |
| 3 | 0.282 m | 0.205 m | 0.768 m | 68.9 mm | 0.232 m | 83 |

**Drift does not grow with chain length** (0.281 → 0.252 → 0.282) while the ORACLE's drift
does (0.113 → 0.205), so what accumulates is tokenizer reconstruction, not chaining. The gap
to the oracle actually *narrows* (0.168 → 0.077). Seam holds at ~69-72 mm across chain
lengths, matching the one-seam probe. **CLAUDE.md risk #1 is retired for N≤3.** N>3 is
untested — HUMANISE clips are too short to cut further, which is itself the limitation.

### Free rollout — multi-metre paths, and the scene arm finally measurable

Waypoints sampled on free floor, spaced in a **band** at the trained displacement scale.
10 segments, 30 paired rollouts (identical waypoints per arm):

| | collision | goal err / seg | seam | path |
|---|---|---|---|---|
| full (with occupancy) | 2.07% | 0.883 m | 79.8 mm | 7.67 m |
| noscene | 2.61% | 1.329 m | 92.2 mm | 7.48 m |
| **straight-line control** | **1.09%** | — | — | 9.25 m commanded |

**Neither model avoids obstacles — both collide MORE than walking the waypoint polyline
directly.** Passive occupancy conditioning does not confer avoidance. This is the argument
for step 10 (collision-guided decoding), and the first time that component has had a
justification measured rather than assumed.

Scene conditioning does beat its ablation on all three metrics, and this is the first setting
where any difference appeared at all — vindicating "chaining is the unlock" (§8). But **one
training run per arm**, so it cannot be separated from seed variance. Do not quote it as
established without a reseed.

**Waypoint spacing is load-bearing and was got wrong first.** Sampling "anywhere beyond
min_step" gives ~4 m hops; the model, trained on 0.63 m displacement, walks ~1.3 m and stops,
yielding 2.5 m goal error that says nothing about chaining. Goals must be spaced at the
trained scale — a band, not a lower bound.

### THE FINDING: goal-following does not generalize to arbitrary goals

This is the most important result in §9 and it qualifies §4 and §5.

| setting | goal err | null (never move) | vs null |
|---|---|---|---|
| in-distribution goals (§5) | 0.132 m | 0.627 m | **−78.9%** |
| free waypoints, full | 0.883 m | 0.925 m | **+4.5%** |
| free waypoints, full, real clip text | 0.981 m | 0.925 m | +6.1% |
| free waypoints, noscene | 1.329 m | 0.925 m | −43.7% (worse than not moving) |

**On arbitrary commanded goals the model performs no better than standing still.** It does
move — 7.7 m of path over 10 segments — but not toward the goal.

Not a text problem: substituting the clip's real utterance for the generic
"walk to the target" changes nothing (0.981 vs 0.883 m). **It is the goals.**

The likely mechanism: the model learned the joint distribution of (motion | text, prefix,
scene, goal) from data in which the goal is *correlated with* the motion that follows. On
goals drawn from that distribution — real clip endpoints — this is indistinguishable from
goal-following, and produced §4's +0.90/+0.97 commanded-vs-achieved correlation. On goals
sampled independently of the data, the correlation is gone and so is the behaviour.

**Consequences, and they are not small:**
- Done-criterion 3 ("single segment reaches an explicit goal") holds only for
  in-distribution goals. Say so when quoting it.
- **Step 11 is directly threatened.** An MLLM planner emits arbitrary coordinates; on this
  evidence they would not be followed. Test goal-following on planner-like goals before
  building the MLLM stage, not after.
- The fix is a training-data question, not an architecture one: the model has never seen a
  goal that was not the endpoint it was already going to reach. Candidate remedies —
  goal augmentation (perturb or resample goals during training), or an explicit trajectory
  objective that penalizes distance-to-goal rather than only token likelihood.

### Other limits

N>3 chaining untested (HUMANISE clips cannot be cut further). Free-waypoint rollouts use
`walk` clips only. One training run per arm throughout.

## 10 · Goal augmentation — DONE, fixes §9's blocking problem

§9 found the model followed familiar goals but not arbitrary ones. Hypothesis: it has only
ever seen the goal it was *already* walking to, so it learned to produce plausible motion
rather than to go somewhere.

**Truncation augmentation** (`train_probe --goal-aug 0.5`): cut the token sequence at a random
length L and set the goal to the world position at frame 4L−1, so goal and target motion are
truncated **together**. One clip then supplies many (goal, motion) pairs at varied distance
and direction.

**The obvious alternative is wrong and is recorded so nobody tries it:** randomising the goal
while keeping the same target motion teaches the model to IGNORE the goal, since that motion
would then be correct for every goal.

### Result — arbitrary goals, 30 rollouts × 10 chained segments

| | goal err | vs 0.925 m null | path vs commanded |
|---|---|---|---|
| step-8 model | 0.883 m | +4.5% | 7.67 / 9.25 m |
| **+ goal augmentation** | **0.374 m** | **−59.6%** | **9.27 / 9.25 m** |

Goal error more than halves, and path length now matches the commanded distance almost
exactly — it walks as far as it is told, which it previously did not.

### No regression in-distribution

| | seam | goal |
|---|---|---|
| step-8 model | 71.7 mm | 0.0584 m |
| + goal augmentation | 73.7 mm | **0.0560 m** |
| ORACLE | 70.9 mm | 0.0544 m |

Familiar-goal accuracy is unchanged (marginally better); seam is 2 mm worse, within noise.
The fix is targeted — it bought out-of-distribution goal-following without costing anything.

### What it did NOT fix, as expected

Collision 2.10% vs 2.07% before, against the 1.09% straight-line control. Augmentation
addresses *where the model goes*, not *what it goes around*. Obstacle avoidance remains open
and is step 11's job.

**Remaining gap:** 0.374 m on arbitrary goals vs 0.132 m on familiar ones. Better, not equal.
Candidate next moves if it needs to close further: raise `--goal-aug` above 0.5, or add an
explicit distance-to-goal training objective rather than relying on token likelihood alone.

## 11 · Interaction in a chain — DONE. Done-criteria 4 and 5 MET

The goal of step 11 (done-criteria 4/5): a chained rollout that actually INTERACTS — walk to
furniture, sit, stand, walk away — not a walk-only path. Reaching it required finding and
fixing two obstacles that every prior metric had hidden, then a `full_action` retrain. The
watchable `walk→sit→stand→walk` clips are at `~/wander_data/step11_demo_multiseg/` (best:
`demo_00` scene0151/couch and `demo_05` scene0694/coffee-table, both 0% collision). Best model
`~/wander_data/step11/checkpoints/action` — supersedes step-10 for interaction.

### The measurement that unblocked it: pelvis height, not goal error

Step 11a first reported sit/stand "work" by **goal error** — 0.14 m for sit vs a 0.18 m null.
That is the §5#4 trap a sixth time: the goal is only (x, y), so **walking to the seated
pelvis's xy scores well without ever sitting**. Goal error cannot see the z-axis, and sitting
is a z-axis event. The composed demo made it visible — 0/3 (step-10) and 0/5 (step-8) chains
sat, pelvis held at ~0.92 m through the "sit" segment while goal error stayed ~0.4 m.

The diagnostic that replaces goal error here is **pelvis height**, oracle-verified through the
identical 263→recover→Z-up path the decoder uses: GT sit drops 0.95→**0.52–0.58 m**, GT
stand-up rises to **~0.98 m**, GT lie ends **~0.1 m**, GT walk holds **~0.95 m** throughout.
The thresholds sit-z<0.70 / stand-z>0.80 separate seated from standing cleanly.

### Obstacle 1 — goal augmentation (§10) suppresses interaction

Single-segment generation from the clip's real (standstill) prefix, measured by pelvis height,
60 clips/action whose GROUND TRUTH interacts:

| model | sit rate | stand rate |
|---|---|---|
| step-8 (no goal-aug) | **75%** | 62% |
| step-10 (+ goal-aug 0.5) | **42%** | 77% |

Goal augmentation trains pure xy-reaching, and that taught the model to walk-to-the-xy-and-stay-standing
instead of sitting: sit 75%→42%. (Stand-up "improving" is the same standing bias passing
trivially.) **Goal error was identical across the two (0.16 vs 0.18 m)** — the proof it is blind.

### Obstacle 2 — the walk→sit SEAM is out of distribution

HUMANISE sit clips barely move (median start→sit **0.11 m**, p90 0.50 m, 0% ≥1.0 m) — the
person is already at the furniture and starts from a **standstill**. So there are **no walk→sit
transitions in the data**, and the seam a chain needs (a mid-stride walking body → sit down) was
never trained. Isolated cleanly (step-8, sit clips, only the prefix varied):

| prefix handed to the sit segment | sit rate |
|---|---|
| standing-still (own frame-0, in-distribution) | **70%** |
| mid-stride walking (OOD) | **0%** |

A walking prefix collapses sitting to zero — the model just keeps walking. This, not the model's
inability to sit, is why every composed walk→sit chain failed.

### The fix — explicit action + a synthesized walk→sit seam (training-time only)

Three changes in `train_probe.py`, no re-tokenization (step-10 tokens already carry `action`,
`prefix_pose`, `occ_crop`):
1. **`cond_mode=full_action`** — a 4-way action one-hot (walk/sit/stand up/lie) appended raw to
   the cond vector (1364→1368). The explicit "sit now" signal the (x,y) goal cannot carry.
2. **`--walk-prefix-aug 0.5`** — for interaction clips, swap in a real mid-stride walking prefix
   (from walk clips' own seam poses) half the time. Synthesizes the missing walk→sit seam:
   *walking body + action=sit → sit tokens*.
3. **`--goal-aug` restricted to WALK clips** — truncation no longer corrupts interaction goals.

### Result — final `full_action` model (20k iters), pelvis-height, 60 clips/action

| prefix | sit rate | stand rate | goal-err |
|---|---|---|---|
| own (standstill) | **83%** | 98% | 0.12 m |
| **walk (was 0%)** | **85%** | 98% | 0.15 m |
| step-10 goalaug (own, for reference) | 42% | 77% | 0.16 m |

Walking-prefix sit **0%→85%** (obstacle 2 solved), standstill sit 42%→**83%** (obstacle 1
solved), stand-up 77%→98%, and goal error did not regress (0.12 vs 0.16 m — it improved). The
one-hot action is the load-bearing addition; the walk-prefix augmentation is what makes it fire
from a walking body.

### The composed demo — done-criteria 4 and 5

`demo_interaction.py`, seeded from real sit clips (verified furniture + known sit target),
`walk→sit→stand→walk` with explicit per-segment actions, pelvis-height SAT/STOOD structure
check (no oracle exists for a composed chain — CLAUDE.md risk #4):

| walk-up | chains SAT **and** STOOD |
|---|---|
| single long segment (start_dist 3 m) | 1 / 8 |
| **multi-segment (auto-split hops, start_dist 3 m)** | **5 / 10** |

**Why the walk-up had to be split.** A single 3 m walk segment UNDERSHOOTS — the model, trained
on 0.63 m mean displacement, walks ~1.3 m and stops (RESULTS §9), leaving the sit segment a
multi-metre goal. And a long sit goal makes the model WALK, not sit: HUMANISE sit clips barely
move, so `action=sit` is entangled with a ~0.1–0.5 m goal, and a longer one is out of
distribution. Splitting the walk-up into ≤1.1 m hops DELIVERS the body to the furniture, so the
sit goal stays short and in distribution. Best chains are at 0% collision; the walk→sit and
sit→stand seams run 300–440 mm (the body settling from a stride into a seat — expected, hidden
by the display blend), while the walk-hop seams stay clean (28–92 mm, matching §7/§9).

**Honest gap.** End-to-end composed SAT is ~50%, below the 85% single-segment capability — a
composed chain compounds (deliver to furniture × fire the sit), and the sit segment's *generated*
walk-end prefix differs from the mid-stride poses `--walk-prefix-aug` trained on. 50% is enough
to harvest a watchable demo; closing it (end-of-walk prefixes in the augmentation, or a higher
`--walk-prefix-aug`) is the next refinement if a higher yield is needed. **Single seed / one
training run, as everywhere in this project.**

### Heading — the body did not turn to face travel; fixed at inference (re-orient), NOT by conditioning

`diag_heading.py`, 6-segment free-waypoint chains: the body held a nearly fixed facing
(circular-std ~22° over a whole chain, ~28° turn within a segment) while travel spanned ±180°,
so it reached side/behind goals by **strafing or walking backward** — |facing − travel| **78°**.

**What was tried and FAILED — target-heading conditioning.** Added `cond_mode=full_action_head`:
an explicit target heading (body yaw at the segment end, sin/cos relative to start),
supervised on GT, commanded = direction-to-goal at inference. At 2k iters it helped (78°→56°),
but the full 20k model **reverted to 76°** — it learned to IGNORE the heading input. Reason:
GT people face their travel (median 8°, checked in `add_goal_heading.py`), so the target is
**predictable from goal+prefix**; a converged 97%-accuracy model reproduces the GT tokens
without needing the redundant input, and it gets no gradient. Using body-yaw instead of
travel-direction did not escape the redundancy. Recorded so nobody re-adds it expecting a win.

**Root cause is geometric, not a missing signal.** The body can only turn ~29° inside one short
HUMANISE segment; free chains demand sharp (>29°) turns between segments, so facing lags travel.
On smooth / GT-like paths the model already faces travel — the moonwalk is an artifact of
sharp waypoint-to-waypoint direction changes.

**What FIXED it — inference re-orient** (`rollout(..., reorient=True)`): rotate each walk
segment's start heading to face its goal, so travel is "forward" every segment. |facing −
travel| **78°→3°**, chain facing spread 22°→**87°** (it now turns to follow the path). It does
NOT break the seam — the prefix pose is heading-canonicalized (frame-independent), so this only
rotates the body about its root; limbs do not teleport. Interaction segments (short goal) keep
their inherited facing so they do not spin in place. This is the lesson twice over: when a
conditioning signal is redundant with what the model already predicts, it is ignored — reach
for an inference-time geometric fix instead.

### Open limitation — the model does not know which way the furniture faces (sit orientation)

The model performs a sit without regard to the furniture's orientation, so it can sit facing the
wrong way. `diag_sit_facing.py` (40 sit clips, `head_action`):
- baseline |sit facing − GT seated facing| = **31°** — not random (90° would be), because the
  sit FOLLOWS THE APPROACH DIRECTION (it sits ~31° off the way it walked in). In a real clip the
  approach is correct, so the sit lands roughly right; with a synthesized approach it does not.
- commanded sit facing is **IGNORED**: |gen(cmd=GT) − gen(cmd=GT+180°)| = **14°** (should be
  ~180° if obeyed). Same as the walk heading — the `full_action_head` heading input is dead at
  convergence, for sit too.

**Root cause: there is no furniture-orientation signal anywhere the model sees.** The occupancy
raster is a FOOTPRINT — a sofa's rectangle says nothing about which side is the seat vs the back.
So the model cannot infer facing, and the one input that could carry it is ignored.

**Rejected as a HACK (recorded so nobody ships it):** arranging the demo so the agent APPROACHES
from the seed clip's GT seated direction makes the sit come out right (sit-facing error ~5°) —
but that peeks at the ground-truth answer, does not touch the model, and does not generalize to a
novel scene (no GT there). Do not do this and call it a result.

**A proper fix is open work:** an orientation-aware scene representation the model actually
consumes (the RGB render shows a chair's front; an oriented-object map from annotations would
too), or forcing the model to depend on an explicit orientation input (the current heading input
is treated as redundant and zeroed). Until then, wrong sit-orientation stands as a known
limitation, honestly.

**Where SOTA is (SceMoS, CVPR 2026):** even the SOTA does NOT explicitly control interaction
orientation — it emerges from the planned approach, same as us. What it DOES have is a
geometry-grounded TOKENIZER (local heightmap → VQ-VAE decoder + contact loss) giving
contact-correct interaction, which we lack (our tokenizer is scene-blind, scene is a
transformer-side footprint). So the near-term orientation answer is placement-side perception;
the deeper interaction upgrade is a heightmap tokenizer. See docs/IN_FLIGHT.md "Next direction".

### In-distribution regression check (`eval_accumulation.py --by-action`, 300 clips)

The fix is not free, and the cost is where you'd expect. Navigation is untouched: N=1 walk
drift 0.366 m and N=2 walk seam **60.7 mm** match step-10 (0.354 m, 59.1 mm), and interaction
goal error *improved* (N=1 sit 0.108 vs 0.139, stand-up 0.182 vs 0.220). What rose is the
**interaction seam** — N=2 sit 68.6→**92.5 mm**, stand-up 112→**154.7 mm** — because
`--walk-prefix-aug` deliberately trains the interaction segment to produce its pose from a
*walking* prefix, i.e. to snap rather than continue. That is the seam the display blend is for,
and the trade bought the walk→sit seam existing at all.

---

## 12 · Geometry-grounded tokenizer (SceMoS port) — BUILT & MEASURED; marginal on our data

The SceMoS-style upgrade (§8/IN_FLIGHT): put scene geometry in the TOKENIZER's decoder (local
heightmap → decoder + contact loss) so decoded interaction is contact-correct, not just
scene-grounded on the transformer side. Fully built (`src/scene_*.py`, `src/contact_loss.py`,
`scripts/scene_tokenizer/*`) and measured end-to-end. **Verdict: the mechanism works, but on our
data it gives only a marginal contact improvement at a real motion-quality cost — it does NOT
reproduce SceMoS's clean contact-correctness (Contact 0.98 on TRUMANS).** A negative-but-useful
result: do not build further on it without new evidence (the DINOv2 precedent, §6).

**Input is real (blocker passed).** ±0.6 m / 32×32 body-frame heightmaps from the ScanNet mesh +
world track capture seats/beds/floor (clearance separates on-surface from above-surface). Two
convention bugs caught by oracles: 263↔track frame alignment (offset 0), and the **vertical
reference** — the heightmap must be CLIP-FLOOR referenced (else a lie-on-bed body reads 0.8 m
"under" the bed, GT penetration 400–770 mm → ~0 after the fix). Contact-loss axis mapping
GT-validated (~0 GT penetration).

**Two approaches, both measured:**
1. **Frozen encoder (cheap).** Heightmap-conditioned decoder only; tokens/transformer reused
   unchanged. A **vertical-shift augmentation** (encode original, shift the heightmap by Δ, require
   the body to shift by Δ — the analog of goal-aug §10) breaks the redundancy trap: reconstruction
   follow-ratio 0.00 → ~1.0. **But it does NOT transfer to generation:** follow=1.0 is the
   reconstruction regime (token consistent with the hm); a *generated* generic sit token carries a
   baked-in nominal height that the frozen decoder respects, so generated seated pelvis is ~const
   0.6 m regardless of the real seat (measured, `eval_contact_demo`).
2. **Unfrozen + shift-consistency (the "proper" fix, user-authorized).** Unfreeze encoder+quantizer
   and add a consistency loss `‖enc(motion) − enc(motion+Δ)‖` to make tokens HEIGHT-AGNOSTIC, so
   the heightmap becomes the contact-height source. Re-extract all tokens + retrain the transformer
   (full cascade). Token shift-invariance climbed 0.57→0.72 (plateaued at consist-weight 0.25;
   would need a higher weight, hence worse recons, to reach ~1.0). Transformer retrained to 97.2%
   token acc on the new codebook.

**Generation result (new transformer + new tokenizer, 39 fired sits).** GT reference: a correctly
seated pelvis sits **0.156 m ± 0.076** above the seat surface (hip joint above the cushion), so
|pelvis−seat|≈0.156 is *correct contact*, not overshoot.

| decode | corr(seat) | \|pelvis−seat\| | vs correct (0.156) |
|---|---|---|---|
| flat heightmap (token only) | +0.41 | 0.186 | ~correct |
| **real heightmap** | **+0.59** | 0.229 | +0.07 overshoot |
| (old scene-blind pipeline, §11) | ~+0.1 | — | fixed nominal, no tracking |

So the heightmap **does** make sits track seat height better (corr 0.1→0.6 vs the old fixed-nominal
pipeline, 0.4→0.6 vs the new token-only baseline) — but it adds a ~7 cm overshoot, so **absolute
contact accuracy is a wash**, and the tokens never went fully height-agnostic (0.72), so the
token-only baseline already tracks most of it.

**Cost.** Reconstruction MPJPE regressed across the board (sit 48→74, stand-up 56→74, h3d 55→63):
the invariance constraint degrades ALL generated motion quality, not just contact. The trade —
marginal contact-tracking gain for a broad motion-quality hit — is not worth shipping.

**Why it underdelivers here vs SceMoS.** SceMoS trains its tokenizer from scratch on a large clean
mocap set (TRUMANS); we retrofit a small finetuned tokenizer, forcing an invariance-vs-fidelity
trade our data can't absorb. Real furniture seats also cluster near the nominal the scene-blind
tokenizer already produces, so the headroom is small. **Recommendation: keep the scene-blind
tokenizer (meets done-criteria 4/5); the bigger interaction gap is sit ORIENTATION (§11), not
contact height. Models/tokens preserved under `~/wander_data/scene_tokenizer/`.**

## 13 · Collision-guided decoding — DONE, works. The model now steers around obstacles

The gap this closes (§8, §9): the model FOLLOWS goals but does not STEER. Greedy chained rollouts
collide with the 0.9 m tall-obstacle map ~2–3.6% of the path — as much as or MORE than a straight
line between the same waypoints. Nothing avoids anything. Step 12 adds inference-time scene
steering with NO extra training (`scripts/chaining/collision_guided.py`).

**Mechanism.** `trans.sample(if_categorial=False)` is greedy argmax — deterministic, so rejection
sampling over it is a no-op. Switching to `Categorical(probs)` draws (`if_categorial=True`) gives
PATH diversity to select over. Two selectors, both compared on the SAME scenes/waypoints/seed with
the straight-line polyline as the oracle control (risk #4):
- **reject_chain** (the guaranteed floor, CLAUDE.md 2e): N whole stochastic chains, keep the
  lowest total collision. Global, not reactive; costs N× generation.
- **guided_seg** (the actual guided decoder): at EACH segment, sample N candidates (candidate 0 =
  greedy argmax, so it is never worse than greedy on its own score), decode + SE(2)-place each, keep
  `argmin(goal_err + w·collision)`. Locally reactive (re-plans every ~1 m hop), cheap (N/segment,
  not Nᴺ), and goal_err stays IN the objective so it cannot "avoid" a wall by refusing to move.

**Result — 20 chained rollouts × 6 segments, `step10/checkpoints/goalaug`, n_cand=8, two seeds:**

| mode | seed 0 collision | seed 0 goal | seed 1 collision | seed 1 goal |
|---|---|---|---|---|
| oracle straight-line | 1.57% | — | 3.93% | — |
| greedy (baseline) | 2.06% | 0.186 m | 3.63% | 0.161 m |
| reject_chain (N=8) | 0.85% | 0.204 m | 2.19% | 0.227 m |
| **guided_seg (w=10)** | **0.69%** | **0.093 m** | **2.64%** | **0.110 m** |

- **Both selectors cut collision below greedy on both seeds.** guided_seg at w=10 also beats the
  straight-line oracle on both seeds (0.69<1.57, 2.64<3.93) — i.e. the model now routes AROUND
  obstacles a direct path would clip, the target from CLAUDE.md §12.
- **guided_seg keeps the task; reject_chain sacrifices it.** guided_seg's goal error (0.09–0.11 m)
  is *better* than greedy (best-of-N over the combined objective, with greedy always a candidate);
  reject_chain's rises (0.20–0.23 m) because whole-chain selection favors wandering low-collision
  routes. Compute-matched (both 8× greedy), guided_seg is the better use of the budget.
- **`coll_weight` sweep {2,5,10,20}, both seeds:** collision falls as w rises while goal error stays
  flat (~0.09–0.12 m) — there is no goal-error cost to steering harder on these scenes. Monotone on
  seed 0 (1.14→0.67%); best at w=10 on seed 1 (2.64%). w=10 chosen as the default.
- **Money shot:** `~/wander_data/step12_fig/cg_compare_scene0001_00.png` — greedy walks *through* a
  tall obstacle (16.3%, worse than the 7.8% straight line); guided steers around it to the same
  waypoints (0.0%).

**Honest caveats.** (1) Two seeds, one model, one dataset — the effect is large and reseeds, but the
exact percentages are not published-comparable (generation FID still unreproduced, §CLAUDE.md 3).
(2) The straight-line control's value swings with the waypoint draw (1.57% vs 3.93%), so the robust
claim is the *relative* reduction vs greedy, not an absolute %. (3) This steers the ROOT PATH against
an occupancy footprint; it is orthogonal to the sit-orientation and contact-height gaps (§11, §12),
which it does not touch. (4) An upstream `t2m_trans.sample` bug (unbound `xs` when a stochastic draw
emits EOS at position 0) is worked around in `safe_sample`, not fixed in the shared base.

## 14 · End-to-end VLM demo — DONE. Instruction -> local plan -> steered interacting motion in a mesh

The full inference pipeline (CLAUDE.md 2c), wired and working (`scripts/planner/`):
`instruction + scene image -> qwen3.5:27b plan -> expand -> guided rollout -> full-mesh video`.

**Grounding — the design that made it work (`scene_anchors.py`).** VLMs regress raw coordinates
poorly, so we do NOT ask for coordinates. Low furniture = cells occupied in the 0.12 m map but FREE
in the 0.9 m map (RESULTS §8) — i.e. sit-able surfaces (sofas/beds/low tables/chair seats), with
walls and open floor removed. Connected components of that mask are numbered ANCHORS, overlaid on the
top-down BEV. The VLM (`qwen_plan.py`, ollama `qwen3.5:27b` vision, `format:"json"`, `think:False`)
picks WHICH numbered anchor each segment targets and WHAT action; geometry converts anchor→world xy.
So the MLLM does the grounding (which object = where) and the motion model still never guesses a
location. Output schema: `{"segments":[{"action∈{walk,sit,stand up,lie}, "target": anchor-id|"away",
"why"}]}`.

**Result (scene0151, "I'm tired… go sit and relax on the couch, then get up and head out").**
qwen grounded "the couch" to anchor #1 (correct, both from the legend size and the image) and emitted
`walk #1 → sit #1 → stand up #1 → walk away`. `expand_plan` split it into 9 rollout segments (walk-ups
auto-split into ≤1.1 m hops, stop 0.35 m short of the seat so the sit goal is in-distribution). The
step-11 action model generated it with `guided_seg` steering on the walk segments (§13) and greedy
decoding on the interaction segments. **Verified by pelvis height (risk #4, goal error is z-blind):
sit → 0.56 m (SAT), stand up → 0.95 m (STOOD).** Path 7.2 m, 3.0% collision. Rendered to
`end2end_scene0151_00_0.mp4` in the room mesh; plan image `plan_scene0151_00.png`.

**Caveats (honest).** (1) The VLM is slow (~80 s/plan on the 27B) and single instruction / single
seed here. (2) The sit still lands at the seat EDGE/corner (the §11 sit-placement/orientation
narrowness carries straight through — the VLM plans correctly, the motion model interacts imperfectly).
(3) Grounding is validated on ONE clear case; ambiguous rooms / abstract sub-goals ("get a coffee")
are untested at scale. (4) `qwen3.5:27b` is what is installed; CLAUDE.md specced Qwen3-VL 8B — swap is
one env var (`WANDER_PLANNER_MODEL`). But the pipeline is REAL and end-to-end: language in, interacting
motion in a real room out, no hand-authored plan.

## Conditioning inputs — evidence status

| input | status |
|---|---|
| relative-frame goal | validated (§4, §5) |
| seam pose | validated (§7) |
| occupancy scene | representation validated (§6); on chained rollouts it beats its ablation on collision/goal/seam (§9), but **single seed per arm** — suggestive, not established. The trained conditioning does NOT by itself produce obstacle avoidance (greedy collides ≥ a straight line); decode-time steering (§13) is what closes that, and it is what the demo uses. |
| action one-hot | validated on the final 20k model (§11): the explicit action signal that makes the model sit/stand rather than navigate to the xy. With `--walk-prefix-aug` it lifts walking-prefix sit 0%→85% and standstill sit 42%→83%. Single training run. |

## Bug ledger — five silent convention bugs, same shape each time

Z-up vs Y-up · wrong mean/std in the MPJPE canary · the same normalization error again in the
grounding probe · the 90° SE(2) yaw error · the collision metric scoring the objective as
failure. The last two were caught only after producing published-looking numbers, and the 90°
one inverted a track's conclusion.

The pattern in the misses: each was checked with a metric structurally incapable of detecting
it. Start-error cannot see a rotation; per-frame MPJPE cannot see cumulative drift; a collision
rate cannot see that it is measuring the goal; and (§11, the sixth instance) **goal error
cannot see whether the person sat — the goal is (x, y) and sitting is a z-axis event, so a
model that walks to the seat and stays standing scores as well as one that sits.** The fix each
time is the same: **every pipeline that emits a number gets an oracle control, and the metric
must be able to move when the thing you care about moves.**
