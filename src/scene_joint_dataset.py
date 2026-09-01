"""Joint H3D+HUMANISE dataset for the geometry-grounded VQ-VAE finetune: like
joint_vqvae_dataset but each item is (window_263_normalized, window_heightmap).

HUMANISE clips carry their precomputed per-frame local heightmap (extract_heightmaps.py), cropped
with the SAME random window as the motion. H3D clips have no scene -> a flat-floor heightmap
(zeros). The crop alignment is load-bearing: motion frame t and heightmap frame t must be the same
physical frame (offset 0, verified 2026-08-28), so the crop uses one shared `start`.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from joint_vqvae_dataset import (  # noqa: F401  reuse the tested loaders/splits
    H3D_ROOT, HUMANISE_ROOT, HUMANISE_263_CACHE, _read_ids, load_h3d_split,
)
from scene_heightmap import GRID_N

HUMANISE_HM_CACHE = os.environ.get(
    "WANDER_HUMANISE_HM_CACHE",
    os.path.expanduser("~/wander_data/motion_data/HUMANISE_heightmap_cache"),
)


def load_humanise_split_with_hm(split, window_size=64, cache_263=HUMANISE_263_CACHE,
                                cache_hm=HUMANISE_HM_CACHE, verbose=True):
    """Loads HUMANISE (263, heightmap) pairs for `split`. A clip is kept only if BOTH its 263 and
    its heightmap exist, are finite, length-match, and are >= window_size. Returns
    (motions, heightmaps, stats) index-aligned."""
    ids = _read_ids(f"{HUMANISE_ROOT}/{split}.txt")
    motions, hms = [], []
    n_short = n_nan = n_missing263 = n_missinghm = n_mismatch = 0
    for name in ids:
        idx = int(name)
        p263 = f"{cache_263}/{idx:05d}.npy"
        phm = f"{cache_hm}/{idx:05d}.npy"
        if not os.path.exists(p263):
            n_missing263 += 1
            continue
        if not os.path.exists(phm):
            n_missinghm += 1
            continue
        m = np.load(p263).astype(np.float32)
        h = np.load(phm)  # float16 (T,32,32)
        if m.shape[0] != h.shape[0]:
            n_mismatch += 1
            continue
        if not np.isfinite(m).all():
            n_nan += 1
            continue
        if m.shape[0] < window_size:
            n_short += 1
            continue
        motions.append(m)
        hms.append(h)  # keep float16 to save RAM; cast per __getitem__
    stats = dict(requested=len(ids), loaded=len(motions), short=n_short, nan=n_nan,
                 missing263=n_missing263, missing_hm=n_missinghm, mismatch=n_mismatch)
    if verbose:
        print(f"[HUMANISE+HM:{split}] loaded {stats['loaded']}/{stats['requested']} "
              f"(short={n_short} nan={n_nan} miss263={n_missing263} misshm={n_missinghm} "
              f"mismatch={n_mismatch})")
    return motions, hms, stats


class HMWindowDataset(Dataset):
    """Windowed (motion, heightmap) pairs. If heightmaps is None (H3D), yields a flat-floor
    (zeros) heightmap. Motion is Z-normalized; heightmap is passed through in metres."""

    def __init__(self, motions, heightmaps, mean, std, window_size=64, grid_n=GRID_N):
        self.motions = motions
        self.heightmaps = heightmaps  # list aligned to motions, or None (flat floor)
        self.mean = mean
        self.std = std
        self.window_size = window_size
        self.grid_n = grid_n

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion = self.motions[idx]
        start = random.randint(0, len(motion) - self.window_size)
        w = motion[start:start + self.window_size]
        w = ((w - self.mean) / self.std).astype(np.float32)
        if self.heightmaps is None:
            hm = np.zeros((self.window_size, self.grid_n, self.grid_n), dtype=np.float32)
        else:
            hm = self.heightmaps[idx][start:start + self.window_size].astype(np.float32)
        return w, hm


def _cycle(loader):
    while True:
        for x in loader:
            yield x


class BalancedJointHMLoader:
    """Same balanced H3D:HUMANISE mix as BalancedJointLoader, but yields (motion, heightmap)
    batches. H3D sub-batch gets flat-floor heightmaps; HUMANISE gets real ones."""

    def __init__(self, h3d_ds, hum_ds, batch_size, h3d_frac=0.5, num_workers=4, seed=0):
        n_h3d = max(1, round(batch_size * h3d_frac))
        n_hum = max(1, batch_size - n_h3d)
        self.n_h3d, self.n_hum, self.batch_size = n_h3d, n_hum, n_h3d + n_hum
        g1 = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        self.h3d_loader = DataLoader(h3d_ds, batch_size=n_h3d, shuffle=True,
                                     num_workers=num_workers, drop_last=True, generator=g1,
                                     persistent_workers=num_workers > 0)
        self.hum_loader = DataLoader(hum_ds, batch_size=n_hum, shuffle=True,
                                     num_workers=num_workers, drop_last=True, generator=g2,
                                     persistent_workers=num_workers > 0)

    def __iter__(self):
        h3d_it = _cycle(self.h3d_loader)
        hum_it = _cycle(self.hum_loader)
        while True:
            ma, ha = next(h3d_it)
            mb, hb = next(hum_it)
            yield torch.cat([ma, mb], dim=0), torch.cat([ha, hb], dim=0)
