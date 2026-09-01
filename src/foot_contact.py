"""Foot-contact cleanup (deskating) for generated motion — a training-free, placement-stage
contact projection. Measured motivation (scripts/contact/measure_foot_contact.py): the VQ-VAE round
trip injects ~2.2x the ground-truth foot-skate (56 -> 125 mm/s) and generation ~2.5x (141 mm/s),
while penetration/float are already ~0 in the clip-floor frame. So the productive "contact" defect
on this pipeline is a PLANTED FOOT SLIDING, not contact height (that headroom is small, RESULTS §12).

Design constraints (why this specific method):
  - ROOT-PRESERVING. Only a leg's knee/ankle/foot may move; pelvis, spine, arms, head and the root
    trajectory are untouched. So the grounded goal/placement (RESULTS §4) and every seam are exactly
    preserved -- deskating cannot change goal error.
  - LENGTH-PRESERVING. After pulling the lower leg toward the plant anchor, a forward re-projection
    from the fixed hip restores every bone to its original per-frame length exactly, so the skeleton
    cannot stretch. Where there is no correction (non-contact frames) the re-projection is an exact
    identity.
  - ANCHORED per contact run, ramped at the edges so the correction eases in/out and never snaps
    against adjacent swing frames.

Frame: J is (T,22,3) Z-up with the floor at `floor` (0 in the clip-floor frame that se2_place emits;
pass the scene floor for world-frame joints). SMPL 22-joint legs (hip,knee,ankle,foot).
"""
import numpy as np

LEGS = [(1, 4, 7, 10), (2, 5, 8, 11)]   # (hip, knee, ankle, foot): L then R
J_LFOOT, J_RFOOT = 10, 11
FPS = 20


def _unit(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def detect_foot_contacts(J, h_contact=0.05, v_contact=0.02, floor=0.0):
    """Per-leg (T,) boolean: foot within h_contact of the floor AND horizontal speed < v_contact
    (the standard skate-metric contact gate). Returns [left_mask, right_mask]."""
    out = []
    for (_, _, _, foot) in LEGS:
        h = J[:, foot, 2] - floor
        spd = np.concatenate([[0.0], np.linalg.norm(np.diff(J[:, foot, :2], axis=0), axis=1)])
        out.append((h < h_contact) & (spd < v_contact))
    return out


def _runs(mask):
    """Contiguous True runs of a boolean array as (start, end_inclusive) pairs."""
    runs, i, T = [], 0, len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j + 1 < T and mask[j + 1]:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    return runs


def deskate(J, h_contact=0.05, v_contact=0.02, floor=0.0, ramp=3, knee_w=0.5,
            min_run=2, clamp_pen=False):
    """Remove foot sliding from J (T,22,3) Z-up. Returns a cleaned copy; input untouched.

    For each planted run of a foot, anchor its horizontal position to the run median and pull the
    lower leg (foot & ankle fully, knee by knee_w) toward that anchor, ramped over `ramp` frames at
    each end. A forward re-projection from the fixed hip then restores exact bone lengths. Root,
    pelvis and upper body are never modified."""
    J = np.asarray(J, np.float64).copy()
    for (hip, knee, ankle, foot), mask in zip(LEGS, detect_foot_contacts(J, h_contact, v_contact, floor)):
        l1 = np.linalg.norm(J[:, knee] - J[:, hip], axis=-1)     # per-frame natural lengths
        l2 = np.linalg.norm(J[:, ankle] - J[:, knee], axis=-1)
        l3 = np.linalg.norm(J[:, foot] - J[:, ankle], axis=-1)
        delta = np.zeros((J.shape[0], 2))
        for (a, b) in _runs(mask):
            if b - a + 1 < min_run:
                continue
            anchor = np.median(J[a:b + 1, foot, :2], axis=0)
            w = np.ones(b - a + 1)
            r = min(ramp, (b - a + 1) // 2)
            if r > 0:
                w[:r] = np.linspace(0, 1, r + 1)[1:]
                w[-r:] = np.linspace(1, 0, r + 1)[:-1]
            delta[a:b + 1] = (anchor[None, :] - J[a:b + 1, foot, :2]) * w[:, None]
        pk = J[:, knee].copy();  pk[:, :2] += delta * knee_w
        pa = J[:, ankle].copy(); pa[:, :2] += delta
        pf = J[:, foot].copy();  pf[:, :2] += delta
        nk = J[:, hip] + _unit(pk - J[:, hip]) * l1[:, None]
        na = nk + _unit(pa - nk) * l2[:, None]
        nf = na + _unit(pf - na) * l3[:, None]
        if clamp_pen:
            nf[:, 2] = np.maximum(nf[:, 2], floor)
        J[:, knee], J[:, ankle], J[:, foot] = nk, na, nf
    return J.astype(np.float32)


def foot_metrics(J, h_contact=0.05, v_contact=0.02, floor=0.0):
    """Foot-skate / penetration / float for J (T,22,3) Z-up, floor at `floor`. All lengths mm.
    skate_mm = mean horizontal displacement of a foot while planted (per frame); *FPS => mm/s."""
    T = J.shape[0]
    skate = []
    for (_, _, _, foot) in LEGS:
        h = J[:, foot, 2] - floor
        xy = J[:, foot, :2]
        spd = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1)])
        contact = (h < h_contact) & (spd < v_contact)
        for t in range(1, T):
            if contact[t] and contact[t - 1]:
                skate.append(np.linalg.norm(xy[t] - xy[t - 1]))
    lf = np.minimum(J[:, J_LFOOT, 2], J[:, J_RFOOT, 2]) - floor
    pen = np.maximum(0.0, -lf)
    both_up = lf > h_contact
    return dict(
        skate_mm=float(np.mean(skate) * 1000) if skate else 0.0,
        skate_n=len(skate),
        pen_mean_mm=float(pen.mean() * 1000),
        pen_max_mm=float(pen.max() * 1000),
        float_mm=float(lf[both_up].mean() * 1000) if both_up.any() else 0.0,
        float_frac=float(both_up.mean()),
    )


def bone_length_drift_mm(J0, J1):
    """Max per-bone length change (mm) between two poses of the SAME topology, over the legs —
    the check that deskate preserved lengths. ~0 by construction of the re-projection."""
    d = 0.0
    for (hip, knee, ankle, foot) in LEGS:
        for a, b in [(hip, knee), (knee, ankle), (ankle, foot)]:
            l0 = np.linalg.norm(J0[:, a] - J0[:, b], axis=-1)
            l1 = np.linalg.norm(J1[:, a] - J1[:, b], axis=-1)
            d = max(d, float(np.abs(l0 - l1).max()) * 1000)
    return d
