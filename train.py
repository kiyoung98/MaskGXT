"""Masked discrete-diffusion generative model for crystal structure prediction.

The model jointly generates, via MaskGIT-style iterative unmasking, a Niggli-
reduced lattice (6 quantized parameter tokens), per-atom quantized fractional
coordinates, a space group, and per-atom Wyckoff tokens. Atomic numbers are
given as conditioning. A transformer with QK-normalized attention and SwiGLU
FFNs predicts all streams; losses use ordinal Gaussian soft targets for the
lattice/coordinate bins and cross-entropy for the space-group/Wyckoff streams.

Training uses an online Euclidean-normalizer translation augmentation: with
probability SHIFT_AUG_P, a uniform origin shift r ~ U[0,1)^3 is drawn, the
Euclidean-normalizer translation coset whose fundamental domain contains r is
selected, all fractional coordinates are shifted by r (mod 1), and each atom's
base Wyckoff letter is relabeled by that coset's letter permutation. Coset
representatives and letter permutations are precomputed per space group. Sites
are presented in a canonical electronegativity order with optional intra/inter-
orbit permutation augmentation and sub-bin Gaussian coordinate noise.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------- dataset selection ---------------------------- #
# MAX_N is a module-level constant baked into the model definition below, so it
# must be resolved before the model code runs. We peek at --dataset here (the
# same value is parsed properly in main()) and look up max_n from configs.py.
# Default mp_20 keeps `import train as T` (from sample.py) working without CLI
# args; sample.py sets MASKGXT_DATASET in the environment to select a dataset.
from configs import DATASETS as _DATASETS

DATASETS = _DATASETS  # re-export so `import train as T; T.DATASETS` works


def _peek_dataset() -> str:
    ds = os.environ.get("MASKGXT_DATASET")
    if ds is None:
        for i, a in enumerate(sys.argv):
            if a == "--dataset" and i + 1 < len(sys.argv):
                ds = sys.argv[i + 1]
                break
            if a.startswith("--dataset="):
                ds = a.split("=", 1)[1]
                break
    if ds is None:
        ds = "mp_20"
    if ds not in _DATASETS:
        sys.exit(f"unknown dataset {ds!r}; choose one of {sorted(_DATASETS)}")
    return ds


DATASET = _peek_dataset()

# ----------------------------- constants ---------------------------------- #
K_BINS = 64                  # bins per axis / lattice param
MASK_ID = K_BINS             # mask token id for bin streams (vocab = K_BINS + 1)
VOCAB_SIZE = K_BINS + 1
MAX_N = _DATASETS[DATASET]["max_n"]   # 20 (mp_20, mp_20_ps) or 52 (mpts_52)
N_LATTICE_TOK = 6            # a,b,c,alpha,beta,gamma
LEN_RANGE = (1.5, 30.0)      # lattice length clamp (Angstrom)
ANG_RANGE = (60.0, 120.0)    # lattice angle clamp (degrees) - Niggli reduced
MAX_Z = 95                   # atomic numbers up to 94 -> embedding size 95

# === atom-permutation-symmetry break ===
# Probability that a training crystal gets an intra+inter-orbit permutation
# applied AFTER the canonical electronegativity sort. 0.5 keeps half the batch
# in the strict canonical order (so the model always sees the exact order it
# will be conditioned on at sample time) and washes out within-element order
# dependence on the other half.
ORBIT_PERM_P = 0.5  # full perm (fixed contiguity-preserving orbit-perm)
# Hardcoded Pauling electronegativity table, index = atomic number Z, length
# MAX_Z+1 = 96 (Z 0..95). Source: pymatgen Element(Z).X (Pauling scale).
# Undefined / noble-gas (He, Ne, Ar) -> 0.0 so they sort FIRST deterministically
# at both train and sample time (no NaN). EN is a function of Z only -> available
# at sample time from atomic_numbers alone -> the reproducible positional signal.
PAULING_EN = [
    0.00, 2.20, 0.00, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.00, 0.93,
    1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0.00, 0.82, 1.00, 1.36, 1.54, 1.63,
    1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 2.55, 2.96,
    3.00, 0.82, 0.95, 1.22, 1.33, 1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93,
    1.69, 1.78, 1.96, 2.05, 2.10, 2.66, 2.60, 0.79, 0.89, 1.10, 1.12, 1.13,
    1.14, 1.13, 1.17, 1.20, 1.20, 1.10, 1.22, 1.23, 1.24, 1.25, 1.10, 1.27,
    1.30, 1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00, 1.62, 2.33, 2.02,
    2.00, 2.20, 2.20, 0.70, 0.90, 1.10, 1.30, 1.50, 1.38, 1.36, 1.28, 1.30,
]
assert len(PAULING_EN) == MAX_Z + 1, len(PAULING_EN)
import torch as _torch_en
EN_TABLE_CPU = _torch_en.tensor(PAULING_EN, dtype=_torch_en.float32)

# === online Euclidean-normalizer translation aug ===
# Probability that a given training example is served with a random global
# origin shift instead of the identity. The shift is a continuous uniform
# r ~ U[0,1)^3 drawn online; the coset c whose fundamental domain contains r
# selects the Wyckoff letter permutation perms[c]. 0.0 disables the
# augmentation entirely.
SHIFT_AUG_P = 0.7  # full shift aug

# === sub-bin coord-noise aug ===
# Per-coordinate Gaussian jitter added in fractional units BEFORE
# quantize_frac. Standard deviation is COORD_NOISE_SIGMA_BIN bin-widths
# (1 bin = 1/K_BINS in frac units). 0.30 bin-widths -> sigma ~= 0.0047 in
# frac coords, so the perturbation is well sub-bin (>99% of mass within
# +-1 bin -> at most label-smoothing of the ordinal target). Applied with
# prob COORD_NOISE_P per crystal so half the batch still sees the exact
# quantized centers. Coords are wrapped mod 1 after the jitter (periodic).
COORD_NOISE_P = 0.5
COORD_NOISE_SIGMA_BIN = 0.30
# Precomputed table precompute/normalizer.pt provides, per space group,
# the finite list of Euclidean-normalizer translation cosets:
#   normalizer[sg]["translations"] : list[np.float32 [3]] coset reps
#                                    (index 0 == identity (0,0,0))
#   normalizer[sg]["perms"]        : list[dict[letter,letter]] paired
#                                    with the translations
# Loaded once; train.py never runs symmetry analysis itself. The
# letter -> joint-token map lives in precompute/wyckoff_index.pt.

# === Wyckoff tokenizer streams ===
# Joint (sg, letter) encoding, sizes loaded from precompute/wyckoff_index.pt.
# Embedding-id convention (see precompute_wyckoff.py):
#   Wyckoff: 0=MASK, 1=UNK, raw joint-index t (>=0) -> t+2
#   SG:      0=MASK, 1..230 = space group number (so id == sg)
WYK_MASK = 0
WYK_UNK = 1
WYK_SPECIAL = 2              # number of reserved ids before raw tokens
SG_MASK = 0
N_SG = 230

# === training config ===
D_MODEL = 768
N_LAYERS = 34
N_HEADS = 12                 # head dim 64
DROPOUT = 0.05
# SwiGLU FFN hidden dim: round(2/3 * 4*D_MODEL)=2048 to a multiple of 64.
# Bias-free 3*D_MODEL*FFN_SWIGLU_HIDDEN params/block.
FFN_SWIGLU_HIDDEN = 2048

BATCH_SIZE = 512             # default; override with --batch_size
LR = 4e-4
LR_FLOOR_FRAC = 0.05
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

LOSS_W_LAT = 0.10
LOSS_W_COORD = 1.5
LOSS_W_SG = 0.10             # space-group CE weight
LOSS_W_WYK = 0.30            # Wyckoff CE weight
# === exp #1: continuous sub-bin offset regression ===
# Each bin stream (coord, lattice) gets a regression head predicting the
# continuous position WITHIN the chosen bin, in bin-width units (~[-0.5,0.5]),
# replacing the random uniform jitter used at decode time. The offset is
# supervised (SmoothL1) at UNMASKED valid positions only -- i.e. where the
# input token IS the true bin -- which matches sampling, where the offset is
# always read AFTER the bin is committed.
LOSS_W_COORD_OFF = 0.2       # coord sub-bin offset regression weight
LOSS_W_LAT_OFF = 0.2         # lattice sub-bin offset regression weight
OFF_SMOOTH_BETA = 0.1        # SmoothL1 transition point (bin-width units)

# === sampling config ===
SAMPLE_STEPS = 150
SAMPLE_BATCH_TOTAL = 256     # comps_per_call <= 256
SAMPLE_GUMBEL_TAU0 = 3.0
SAMPLE_TEMPERATURE = 1.0

# === ordinal coord loss ===
COORD_SOFT_SIGMA = 1.0


def _build_coord_soft_targets(k_bins: int, sigma: float) -> torch.Tensor:
    idx = torch.arange(k_bins, dtype=torch.float32)
    d = (idx[None, :] - idx[:, None]).abs()
    d = torch.minimum(d, k_bins - d)            # circular distance
    w = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    w = w / w.sum(dim=-1, keepdim=True)
    return w


COORD_SOFT_TARGETS_CPU = _build_coord_soft_targets(K_BINS, COORD_SOFT_SIGMA)
_COORD_SOFT_CACHE: dict = {}


def coord_soft_targets_on(device) -> torch.Tensor:
    t = _COORD_SOFT_CACHE.get(device)
    if t is None:
        t = COORD_SOFT_TARGETS_CPU.to(device)
        _COORD_SOFT_CACHE[device] = t
    return t


# === ordinal lattice loss ===
# Same Gaussian-soft-target smoothing as the coord loss, applied to the 6
# lattice tokens (a,b,c lengths + alpha,beta,gamma angles), but on a
# NON-CIRCULAR (linear) bin axis: lattice params do NOT wrap, so distance
# is |i-b| (NOT min(|i-b|, K-|i-b|)). Gaussian mass that falls off the
# [0,K) ends is renormalized into the valid range (rows still sum to 1) ->
# a truncated/renormalized Gaussian. One shared sigma (~1 bin, same scale
# as the coord loss) for all 6 lattice positions despite mixed semantics.
LAT_SOFT_SIGMA = 1.0         # Gaussian std in bin units (non-circular)


def _build_lat_soft_targets(k_bins: int, sigma: float) -> torch.Tensor:
    """Row-normalized [k_bins, k_bins] NON-circular (truncated) Gaussian
    soft targets for the lattice tokens. Row b = discretized Gaussian
    centered at bin b, distance measured on the LINEAR bin axis (|i-b|, no
    wraparound). Mass off the [0,k_bins) ends is folded back in by the
    renormalization, so rows still sum to 1 and edge bins carry slightly
    more self-mass. Central term is always > 0, so rows are strictly
    positive at the true bin."""
    idx = torch.arange(k_bins, dtype=torch.float32)
    d = (idx[None, :] - idx[:, None]).abs()     # LINEAR distance (no wrap)
    w = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    w = w / w.sum(dim=-1, keepdim=True)         # truncated -> renormalized
    return w


LAT_SOFT_TARGETS_CPU = _build_lat_soft_targets(K_BINS, LAT_SOFT_SIGMA)
_LAT_SOFT_CACHE: dict = {}


def lat_soft_targets_on(device) -> torch.Tensor:
    t = _LAT_SOFT_CACHE.get(device)
    if t is None:
        t = LAT_SOFT_TARGETS_CPU.to(device)
        _LAT_SOFT_CACHE[device] = t
    return t


# --------------------- Wyckoff/SG precompute loading ---------------------- #
def load_wyckoff_artifacts(script_dir: Path, dataset: str = DATASET):
    """Load the precomputed global (sg,letter) index + per-crystal cache.

    The (sg,letter) index is dataset-independent (one file); the per-crystal
    cache is dataset-specific (wyckoff_cache_<dataset>.pt).

    Returns (index_dict, cache_list, W_RAW). Fails fast if the precompute
    artifact is missing — train.py must NOT run the precompute itself.
    """
    pc = script_dir / "precompute"
    idx_path = pc / "wyckoff_index.pt"
    cache_path = pc / f"wyckoff_cache_{dataset}.pt"
    if not idx_path.exists() or not cache_path.exists():
        sys.exit(
            f"[fatal] missing Wyckoff precompute artifacts under {pc} "
            f"(need wyckoff_index.pt and wyckoff_cache_{dataset}.pt). "
            f"Run `python precompute_wyckoff.py --dataset {dataset}` first."
        )
    index = torch.load(idx_path, weights_only=False)
    cache = torch.load(cache_path, weights_only=False)
    w_raw = len(index)
    assert 1500 <= w_raw <= 2000, f"unexpected Wyckoff vocab {w_raw}"
    return index, cache, w_raw


def load_normalizer(script_dir: Path, wyk_index: dict):
    """Load the per-SG Euclidean-normalizer translation cosets and bake them
    into compact per-SG tensor blocks suitable for fast online sampling.

    Returns a `NormalizerTable` namespace with three per-SG-indexed lookups:
      * `trans[sg]`     : torch.float32 [Cs, 3] coset rep translations
                          (Cs >= 1, index 0 == identity (0,0,0))
      * `letter_map[sg]`: torch.int16 [Cs, 27] mapping base-letter index
                          0..26 ('a'..'aa' -- pyxtal uses single ASCII
                          letters up to 'z' plus a few; we use lowercase
                          'a' (=0) through 'z' (=25), with index 26
                          reserved as -1/UNK for any unmapped letter, which
                          should never happen on the train split). Each
                          row is a permutation among the letters actually
                          present in the SG; absent letters map to -1.
      * `letter_to_tok[sg]`: torch.int32 [27] base-letter index -> raw
                          joint Wyckoff token id (>=0). Absent letters
                          map to -1 / UNK so the dataset can fall back.

    Falls back to a single-identity-coset table per SG (NO augmentation
    effect) if the artifact is missing -- the run never crashes for want
    of the precompute, logged loudly.
    """
    path = script_dir / "precompute" / "normalizer.pt"
    cache: dict[int, dict] = {}
    if not path.exists():
        print(f"[aug] WARNING normalizer.pt missing under {path}; "
              f"DISABLING online normalizer augmentation (identity only).",
              flush=True)
    else:
        try:
            cache = torch.load(path, weights_only=False)
            assert isinstance(cache, dict) and len(cache) == 230, (
                "expected dict keyed by sg 1..230, got "
                f"{type(cache).__name__} len={len(cache) if hasattr(cache, '__len__') else None}"
            )
        except Exception as e:
            print(f"[aug] WARNING normalizer.pt unusable "
                  f"({type(e).__name__}: {e}); disabling aug.", flush=True)
            cache = {}

    L_MAX = 27   # 'a'..'z' plus 1 sentinel slot
    def li(c: str) -> int:
        # single ASCII lowercase letter -> 0..25; anything else -> -1
        if len(c) == 1 and "a" <= c <= "z":
            return ord(c) - ord("a")
        return -1

    trans: dict[int, torch.Tensor] = {}
    letter_map: dict[int, torch.Tensor] = {}
    letter_to_tok: dict[int, torch.Tensor] = {}
    n_nontriv = 0
    n_triv = 0
    for sg in range(1, 231):
        # Build the per-SG (sg, letter) -> token lookup from the global index.
        lt = torch.full((L_MAX,), -1, dtype=torch.int32)
        for (s, let), tok in wyk_index.items():
            if s != sg:
                continue
            j = li(let)
            if 0 <= j < L_MAX - 1:
                lt[j] = int(tok)
        letter_to_tok[sg] = lt

        e = cache.get(sg)
        if e is None or len(e.get("translations", [])) <= 1:
            # Trivial: only the identity coset. No relabel, no shift effect
            # beyond a continuous origin shift (still valuable since the
            # joint token is unchanged for these SGs).
            n_triv += 1
            trans[sg] = torch.zeros(1, 3, dtype=torch.float32)
            lm = torch.full((1, L_MAX), -1, dtype=torch.int16)
            for j in range(L_MAX - 1):
                if int(lt[j]) >= 0:
                    lm[0, j] = j
            letter_map[sg] = lm
            continue

        ts = e["translations"]
        ps = e["perms"]
        Cs = len(ts)
        assert len(ps) == Cs, (sg, Cs, len(ps))
        n_nontriv += 1
        tr = torch.zeros(Cs, 3, dtype=torch.float32)
        lm = torch.full((Cs, L_MAX), -1, dtype=torch.int16)
        for c in range(Cs):
            t = np.asarray(ts[c], dtype=np.float32)
            assert t.shape == (3,), (sg, c, t.shape)
            tr[c] = torch.from_numpy(t.copy())
            for src_let, dst_let in ps[c].items():
                j = li(src_let); k = li(dst_let)
                if 0 <= j < L_MAX - 1 and 0 <= k < L_MAX - 1:
                    lm[c, j] = k
        # Identity coset must be index 0 with zero translation; assert.
        assert torch.all(tr[0] == 0.0), f"sg{sg} coset0 not identity translation: {tr[0]}"
        for j in range(L_MAX - 1):
            if int(lt[j]) >= 0:
                assert lm[0, j] == j, f"sg{sg} coset0 perm not identity at letter {j}"
        trans[sg] = tr
        letter_map[sg] = lm

    if cache:
        print(f"[aug] online normalizer augmentation loaded: 230 SGs "
              f"({n_nontriv} non-trivial, {n_triv} identity-only) p={SHIFT_AUG_P}",
              flush=True)
    return {"trans": trans, "letter_map": letter_map, "letter_to_tok": letter_to_tok,
            "enabled": bool(cache)}


# ----------------------------- data --------------------------------------- #
def lattice_matrix_to_params(L: torch.Tensor):
    a = L[0].norm()
    b = L[1].norm()
    c = L[2].norm()
    cos_alpha = (L[1] @ L[2]) / (b * c + 1e-12)
    cos_beta = (L[0] @ L[2]) / (a * c + 1e-12)
    cos_gamma = (L[0] @ L[1]) / (a * b + 1e-12)
    cos_alpha = cos_alpha.clamp(-1 + 1e-7, 1 - 1e-7)
    cos_beta = cos_beta.clamp(-1 + 1e-7, 1 - 1e-7)
    cos_gamma = cos_gamma.clamp(-1 + 1e-7, 1 - 1e-7)
    alpha = torch.rad2deg(torch.acos(cos_alpha))
    beta = torch.rad2deg(torch.acos(cos_beta))
    gamma = torch.rad2deg(torch.acos(cos_gamma))
    return torch.stack([a, b, c, alpha, beta, gamma])


def params_to_lattice_matrix(params: np.ndarray):
    a, b, c, alpha, beta, gamma = params.tolist()
    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)
    ax = a
    bx = b * math.cos(gamma_r)
    by = b * math.sin(gamma_r)
    cx = c * math.cos(beta_r)
    cy = c * (math.cos(alpha_r) - math.cos(beta_r) * math.cos(gamma_r)) / max(math.sin(gamma_r), 1e-8)
    cz_sq = max(c * c - cx * cx - cy * cy, 1e-8)
    cz = math.sqrt(cz_sq)
    return np.array([[ax, 0.0, 0.0],
                     [bx, by, 0.0],
                     [cx, cy, cz]], dtype=np.float64)


def quantize_lat_params(params: torch.Tensor) -> torch.Tensor:
    out = torch.empty(6, dtype=torch.long)
    for i in range(3):
        lo, hi = LEN_RANGE
        v = params[i].clamp(lo, hi - 1e-6)
        out[i] = ((v - lo) / (hi - lo) * K_BINS).long().clamp(0, K_BINS - 1)
    for i in range(3, 6):
        lo, hi = ANG_RANGE
        v = params[i].clamp(lo, hi - 1e-6)
        out[i] = ((v - lo) / (hi - lo) * K_BINS).long().clamp(0, K_BINS - 1)
    return out


def dequantize_lat_params(tokens: np.ndarray, rng: np.random.Generator | None = None,
                          offset: np.ndarray | None = None) -> np.ndarray:
    # offset (if given) is the model-predicted sub-bin position in bin-width
    # units (~[-0.5,0.5]); it REPLACES the random sub-bin jitter (exp #1).
    out = np.empty(6, dtype=np.float64)
    for i in range(3):
        lo, hi = LEN_RANGE
        step = (hi - lo) / K_BINS
        center = lo + (tokens[i] + 0.5) * step
        if offset is not None:
            center += float(offset[i]) * step
        elif rng is not None:
            center += (rng.random() - 0.5) * step
        out[i] = center
    for i in range(3, 6):
        lo, hi = ANG_RANGE
        step = (hi - lo) / K_BINS
        center = lo + (tokens[i] + 0.5) * step
        if offset is not None:
            center += float(offset[i]) * step
        elif rng is not None:
            center += (rng.random() - 0.5) * step
        out[i] = center
    return out


def lat_params_offset(params: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Sub-bin offset target (bin-width units, ~[-0.5,0.5)) of each of the 6
    lattice params within its quantized bin: continuous normalized position
    minus the bin centre. Mirror of quantize_lat_params' binning."""
    off = torch.empty(6, dtype=torch.float32)
    for i in range(3):
        lo, hi = LEN_RANGE
        v = params[i].clamp(lo, hi - 1e-6)
        vn = (v - lo) / (hi - lo) * K_BINS
        off[i] = vn - (tokens[i].float() + 0.5)
    for i in range(3, 6):
        lo, hi = ANG_RANGE
        v = params[i].clamp(lo, hi - 1e-6)
        vn = (v - lo) / (hi - lo) * K_BINS
        off[i] = vn - (tokens[i].float() + 0.5)
    return off.clamp_(-0.5, 0.5)


def quantize_frac(coords: torch.Tensor) -> torch.Tensor:
    q = (coords % 1.0) * K_BINS
    return q.long().clamp(0, K_BINS - 1)


def dequantize_frac(tokens: np.ndarray, rng: np.random.Generator | None = None,
                    offset: np.ndarray | None = None) -> np.ndarray:
    # offset (if given) is the model-predicted sub-bin position in bin-width
    # units (~[-0.5,0.5]); it REPLACES the random sub-bin jitter (exp #1).
    step = 1.0 / K_BINS
    centers = (tokens.astype(np.float64) + 0.5) * step
    if offset is not None:
        centers = centers + offset.astype(np.float64) * step
    elif rng is not None:
        centers = centers + (rng.random(size=centers.shape) - 0.5) * step
    return centers % 1.0


def wyk_to_embed_id(raw_tok: int) -> int:
    """raw joint (sg,letter) index (>=0) -> embed id; -1 (UNK) -> WYK_UNK."""
    return WYK_UNK if raw_tok < 0 else raw_tok + WYK_SPECIAL



class CrystalDataset(Dataset):
    def __init__(self, path: str, wyckoff_cache: list, wyk_index: dict,
                 normalizer: dict | None = None,
                 shift_aug_p: float = 0.0, orbit_perm_p: float = 0.0,
                 coord_noise_p: float = 0.0, coord_noise_sigma_bin: float = 0.0):
        self.records = torch.load(path, weights_only=False)
        self.wyk = wyckoff_cache
        assert len(self.wyk) == len(self.records), (
            f"wyckoff cache len {len(self.wyk)} != train len {len(self.records)}; "
            "precompute was run on a different train split."
        )
        # === online Euclidean-normalizer translation aug ===
        # `normalizer` is the dict returned by load_normalizer(); per SG it
        # holds the small Cs-by-{3, L_MAX} translation + letter-permutation
        # tables. We pre-bake the per-crystal BASE-LETTER index array (0..25
        # for 'a'..'z', -1 for unknown / out-of-vocab tokens) so the hot
        # path in __getitem__ is a few small tensor ops and no Python dict
        # lookups inside the worker loop. Reverse map (sg, raw_tok) ->
        # letter idx is built once here from wyk_index.
        self.normalizer = normalizer
        self.shift_aug_p = shift_aug_p if (normalizer is not None and normalizer.get("enabled")) else 0.0
        rev: dict[tuple[int, int], int] = {}
        for (s_, let), tok in wyk_index.items():
            if len(let) == 1 and "a" <= let <= "z":
                rev[(s_, int(tok))] = ord(let) - ord("a")
        # Per-crystal base letter array (long, length MAX_N, padded -1).
        self.base_letter = torch.full(
            (len(self.records), MAX_N), -1, dtype=torch.long
        )
        n_with_base = 0
        for i, w in enumerate(self.wyk):
            sg_i = int(w["sg"])
            toks = w["wyckoff_tokens"]
            for j, t in enumerate(toks):
                if j >= MAX_N:
                    break
                li = rev.get((sg_i, int(t)), -1)
                self.base_letter[i, j] = li
                if li >= 0:
                    n_with_base += 1
        # Smoke / debug-time sanity: identity coset reproduces base tokens.
        # Cheap O(\#letters) check on a few SGs.
        if normalizer is not None and normalizer.get("enabled"):
            for sg_chk in (1, 2, 139, 225):
                ltt = normalizer["letter_to_tok"][sg_chk]
                lm0 = normalizer["letter_map"][sg_chk][0]   # identity coset row
                for j in range(26):
                    tj = int(ltt[j])
                    if tj >= 0:
                        assert int(lm0[j]) == j, (sg_chk, j, int(lm0[j]))
            print(f"[aug] base-letter cache built: "
                  f"{n_with_base} per-atom base letters resolved across "
                  f"{len(self.records)} crystals (orbit-letter coverage "
                  f"of cached wyckoff_tokens).", flush=True)
        # === atom-permutation-symmetry break ===
        self.orbit_perm_p = orbit_perm_p
        self.en_table = EN_TABLE_CPU
        # === sub-bin coord-noise aug ===
        self.coord_noise_p = coord_noise_p
        self.coord_noise_sigma = float(coord_noise_sigma_bin) / float(K_BINS)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        lat = r["lattice"]
        frac = r["frac_coords"]
        Z = r["atomic_numbers"]
        N = Z.shape[0]
        params = lattice_matrix_to_params(lat)
        lat_tok = quantize_lat_params(params)         # lattice unchanged by shift
        lat_offset = lat_params_offset(params, lat_tok)   # exp#1: sub-bin target

        w = self.wyk[idx]
        sg = int(w["sg"])
        sg_tok = sg if 1 <= sg <= N_SG else SG_MASK   # SG origin-invariant: keep

        # --- online Euclidean-normalizer translation aug ---
        # With prob shift_aug_p: draw r ~ U[0,1)^3, find the coset c whose
        # fundamental domain contains r (= argmin wrapped distance from r to
        # the coset translation reps), shift ALL frac coords by r (mod 1),
        # and apply the coset's letter permutation to each atom's base
        # Wyckoff letter -> raw joint token. Identity coset (c=0) is always
        # the no-op fallback, so SGs with a 1-element normalizer simply get
        # an identity relabel (still a valid continuous origin shift). All
        # work is small fixed-shape tensor ops; no Python symmetry calls.
        use_aug = (self.shift_aug_p > 0.0
                   and self.normalizer is not None
                   and self.normalizer.get("enabled")
                   and torch.rand(()).item() < self.shift_aug_p)
        if use_aug:
            trans_sg = self.normalizer["trans"][sg]           # [Cs, 3]
            lmap_sg = self.normalizer["letter_map"][sg]       # [Cs, L_MAX] int16
            lt_sg = self.normalizer["letter_to_tok"][sg]      # [L_MAX] int32
            # uniform r in [0,1)^3
            r_shift = torch.rand(3)
            # wrapped (torus) distance from r_shift to each coset rep
            d = (r_shift[None, :] - trans_sg) % 1.0
            d = torch.minimum(d, 1.0 - d)
            dist2 = (d * d).sum(dim=-1)                       # [Cs]
            c = int(torch.argmin(dist2).item())
            frac = (frac + r_shift[None, :]) % 1.0
            # Per-atom relabel: base_letter idx -> relabelled letter idx -> token
            bl = self.base_letter[idx, :N].clone()            # [N] long, -1 = UNK
            wyk_ids = torch.full((N,), WYK_UNK, dtype=torch.long)
            row_perm = lmap_sg[c].long()                      # [L_MAX] long
            for i in range(N):
                li = int(bl[i])
                if 0 <= li < 26:
                    lj = int(row_perm[li])
                    if 0 <= lj < 26:
                        tok = int(lt_sg[lj])
                        if tok >= 0:
                            wyk_ids[i] = wyk_to_embed_id(tok)
                # else: keep WYK_UNK
            # cheap runtime guard: (c) coord wrap is mod-1, (a) identity-coset
            # reproduces base tokens (sampled occasionally, near-free).
            assert float(frac.min()) >= 0.0 and float(frac.max()) < 1.0 + 1e-6,                 ("coord wrap failed", float(frac.min()), float(frac.max()))
            if c == 0:
                # identity coset must reproduce base joint tokens
                raw_w0 = w["wyckoff_tokens"]
                for i in range(min(N, len(raw_w0))):
                    base_tok = int(raw_w0[i])
                    base_emb = wyk_to_embed_id(base_tok)
                    # Only check atoms whose letter is in our [0,26) range
                    if 0 <= int(bl[i]) < 26:
                        assert int(wyk_ids[i]) == base_emb,                             ("identity coset relabel mismatch",
                             sg, i, int(wyk_ids[i]), base_emb)
        else:
            raw_w = w["wyckoff_tokens"]
            wyk_ids = torch.full((N,), WYK_UNK, dtype=torch.long)
            for i in range(min(N, len(raw_w))):
                wyk_ids[i] = wyk_to_embed_id(int(raw_w[i]))

        # === sub-bin Gaussian coord-noise aug ===
        # Independent per-coordinate Gaussian jitter in fractional units,
        # applied AFTER the (already-shifted) frac coords, BEFORE quantize.
        # Periodic wrap (mod 1) preserves the exact crystal symmetry that
        # the data prior assumes. Only fires for prob coord_noise_p.
        if self.coord_noise_p > 0.0 and self.coord_noise_sigma > 0.0 \
                and torch.rand(()).item() < self.coord_noise_p:
            noise = torch.randn_like(frac) * self.coord_noise_sigma
            frac = (frac + noise) % 1.0
        coord_tok = quantize_frac(frac)
        # exp#1: sub-bin offset target (bin-width units), aligned with coord_tok
        # BEFORE the permutations below; permuted alongside coord_tok.
        coord_offset = ((frac % 1.0) * K_BINS
                        - (coord_tok.float() + 0.5)).clamp(-0.5, 0.5)
        Z = Z.long()

        # === atom-permutation-symmetry break ===
        # Applied AFTER the frac-shift aug. orbit id (equivalent_atoms) and the
        # wyckoff token are ORIGIN-INVARIANT, so the shift above did not
        # invalidate them; we use them only as sort keys / permutation blocks and
        # never recompute any symmetry here (pure index ops).
        if N > 1:
            w_base = self.wyk[idx]
            ea = w_base["equivalent_atoms"]
            orbit = torch.zeros(N, dtype=torch.long)
            for i in range(min(N, len(ea))):
                orbit[i] = int(ea[i])
            # raw wyckoff joint token (origin-invariant int) for the tiebreak key
            wkey = torch.zeros(N, dtype=torch.long)
            wraw = w_base["wyckoff_tokens"]
            for i in range(min(N, len(wraw))):
                wkey[i] = int(wraw[i])

            en = self.en_table[Z.clamp(0, MAX_Z)]           # [N] float, fn of Z
            orig = torch.arange(N, dtype=torch.long)

            # canonical stable sort: primary EN[Z] (==sample-time key), then
            # wyckoff token, then orbit id, then original index (uniqueness).
            # successive stable sorts on increasing-priority keys == lexsort.
            perm = orig.clone()
            for key in (orig, orbit, wkey):                 # least -> most sig
                perm = perm[torch.argsort(key[perm], stable=True)]
            perm = perm[torch.argsort(en[perm], stable=True)]  # EN is primary

            Z = Z[perm]; coord_tok = coord_tok[perm]; coord_offset = coord_offset[perm]
            wyk_ids = wyk_ids[perm]; orbit = orbit[perm]; en = en[perm]; wkey = wkey[perm]  # FIX: align wkey for orbit-perm

            # --- intra + inter-orbit permutation augmentation (mcflow port) ---
            # With prob orbit_perm_p, relabel positional slots WITHIN the EN
            # ordering: shuffle whole (EN,wyckoff,orbit) groups (inter) and atoms
            # inside each orbit (intra). Keep the EN-primary order intact so the
            # inter-element sequence the model conditions on at sample time is
            # preserved; only the finer (washed-out) within-element order moves.
            if self.orbit_perm_p > 0.0 and torch.rand(()).item() < self.orbit_perm_p:
                # IMPORTANT: keep the EN-primary inter-element order INTACT (that
                # is the reproducible sample-time signal). The perm only reorders
                # WITHIN each EN group: it shuffles orbit blocks (inter-orbit) and
                # atoms inside each orbit (intra-orbit) -- the within-element order
                # that the sampler cannot reproduce, so we wash it out.
                #   en_rank: contiguous id of the EN group (0,1,2,... by EN value)
                # FIX: hierarchical lexsort (NOT an additive key) so orbit_prio is
                # strictly more significant than intra_noise -> atoms NEVER leave
                # their orbit (orbit contiguity preserved), matching mcflow.
                #   primary EN (electronegativity) -> (element,wyckoff) group ->
                #   per-orbit prio (INTER: reorder whole orbits) ->
                #   per-atom noise (INTRA: shuffle within an orbit).
                _, orbit_block = torch.unique(orbit, return_inverse=True)
                orbit_prio = torch.rand(int(orbit_block.max()) + 1)[orbit_block]  # per-orbit, constant within orbit
                intra_noise = torch.rand(N)                                       # per-atom
                pos = torch.arange(N, dtype=torch.float32)
                pp = torch.arange(N)
                pp = pp[torch.argsort(intra_noise[pp] + 1e-9 * pos[pp], stable=True)]  # intra (within orbit)
                pp = pp[torch.argsort(orbit_prio[pp], stable=True)]                    # inter (orbit order; contiguous)
                pp = pp[torch.argsort(wkey[pp], stable=True)]                          # (element,wyckoff) group preserved
                pp = pp[torch.argsort(en[pp], stable=True)]                            # electronegativity primary
                Z = Z[pp]; coord_tok = coord_tok[pp]; wyk_ids = wyk_ids[pp]
                coord_offset = coord_offset[pp]

        Z_final = Z.long()
        return {
            "lat_tok": lat_tok,
            "coord_tok": coord_tok,
            "coord_offset": coord_offset,
            "lat_offset": lat_offset,
            "Z": Z_final,
            "sg_tok": sg_tok,
            "wyk_tok": wyk_ids,
            "N": N,
        }


def collate(batch):
    B = len(batch)
    N_max = max(b["N"] for b in batch)
    lat_tok = torch.stack([b["lat_tok"] for b in batch])
    lat_offset = torch.stack([b["lat_offset"] for b in batch])   # (B,6) exp#1
    coord_tok = torch.zeros(B, N_max, 3, dtype=torch.long)
    coord_offset = torch.zeros(B, N_max, 3, dtype=torch.float32)  # exp#1
    Z = torch.zeros(B, N_max, dtype=torch.long)
    wyk_tok = torch.zeros(B, N_max, dtype=torch.long)
    sg_tok = torch.tensor([b["sg_tok"] for b in batch], dtype=torch.long)
    site_mask = torch.zeros(B, N_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["N"]
        coord_tok[i, :n] = b["coord_tok"]
        coord_offset[i, :n] = b["coord_offset"]
        Z[i, :n] = b["Z"]
        wyk_tok[i, :n] = b["wyk_tok"]
        site_mask[i, :n] = True
    return {
        "lat_tok": lat_tok,
        "lat_offset": lat_offset,
        "coord_tok": coord_tok,
        "coord_offset": coord_offset,
        "Z": Z,
        "sg_tok": sg_tok,
        "wyk_tok": wyk_tok,
        "site_mask": site_mask,
    }


# ----------------------------- model -------------------------------------- #
class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
        ang = t[:, None] * freqs[None] * 2 * math.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with QK-normalization. Attention is an
    explicit q/k/v projection + F.scaled_dot_product_attention (fused flash /
    mem-efficient kernel, EAGER-available on Py3.13 where compile is off).

    QK-norm: q and k are L2-normalized along head_dim before the dot product
    (Dehghani et al. ViT-22B, arXiv:2302.05442; standard in many recent LLMs).
    This bounds each pre-softmax logit |q.k| <= ||q||*||k|| = 1, so the raw
    dot product lives in [-1, 1]. A learnable per-head temperature `qk_scale`
    (init log(sqrt(head_dim))) restores expressive logit range and REPLACES
    SDPA's default 1/sqrt(hd) scale (we pass scale=1.0) so the unit-norm
    logits are not double-shrunk into a near-uniform softmax.
    """
    def __init__(self, d=D_MODEL, h=N_HEADS, drop=DROPOUT):
        super().__init__()
        assert d % h == 0, "d_model must be divisible by n_heads"
        self.h = h
        self.hd = d // h
        self.drop = drop
        self.ln1 = nn.LayerNorm(d)
        # packed in-projection (q,k,v) like MHA's in_proj_weight, then out_proj
        self.in_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        # SwiGLU gated FFN (Shazeer 2020): FFN(x) = w2( SiLU(w1 x) * (w3 x) ).
        # Hidden H = round(2/3 * 4d) to a multiple of 64 -> 2048 for d=768,
        # the canonical 2/3 sizing. Bias-free -> 3*d*H params/block. Default
        # (Kaiming-uniform) init on all three Linears.
        self.ffn_hidden = FFN_SWIGLU_HIDDEN
        self.w1 = nn.Linear(d, self.ffn_hidden, bias=False)  # gate branch
        self.w3 = nn.Linear(d, self.ffn_hidden, bias=False)  # value branch
        self.w2 = nn.Linear(self.ffn_hidden, d, bias=False)  # down proj
        # Learnable per-head temperature for the QK-normed dot product.
        # Init to log(sqrt(hd)): exp(.) = sqrt(hd) ~ 8 for hd=64, so a
        # fully-aligned q,k pair (cos=1) gives a logit ~8 -- a healthy,
        # non-degenerate initial softmax. Clamped in forward (no blow-up).
        self.qk_log_scale = nn.Parameter(
            torch.full((h, 1, 1), math.log(self.hd ** 0.5))
        )

    def forward(self, x, key_padding_mask=None):
        B, S, d = x.shape
        h = self.ln1(x)
        qkv = self.in_proj(h)                       # (B, S, 3d)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, S, d) -> (B, h, S, hd)
        q = q.view(B, S, self.h, self.hd).transpose(1, 2)
        k = k.view(B, S, self.h, self.hd).transpose(1, 2)
        v = v.view(B, S, self.h, self.hd).transpose(1, 2)

        # --- QK-normalization ---
        # L2-normalize q,k along head_dim so each logit element |q.k| <= 1,
        # bounding attention logits. Fold the per-head temperature into q and
        # pass scale=1.0 to SDPA so its built-in 1/sqrt(hd) factor is NOT
        # applied on top (which would double-shrink the unit-norm logits).
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # clamp temperature to a sane band (exp in [~0.14, ~33]) to prevent
        # any runaway logit scale during training.
        scale = torch.exp(self.qk_log_scale.clamp(-2.0, 3.5))  # (h,1,1)
        q = q * scale                                          # (B,h,S,hd)

        # Build boolean SDPA attn_mask: True = key participates, False = masked.
        # key_padding_mask is (B, S) with True = PADDED key. SG + lattice
        # tokens are never padded, so every query row keeps >=1 valid key
        # (no all-False row -> no softmax-NaN).
        attn_mask = None
        if key_padding_mask is not None:
            keep = (~key_padding_mask).view(B, 1, 1, S)   # (B,1,1,S) bool
            attn_mask = keep.expand(B, self.h, S, S)
        dp = self.drop if self.training else 0.0
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dp, scale=1.0
        )                                            # (B, h, S, hd)
        a = a.transpose(1, 2).reshape(B, S, d)
        a = self.out_proj(a)
        x = x + a
        h2 = self.ln2(x)
        x = x + self.w2(F.silu(self.w1(h2)) * self.w3(h2))
        return x


class MaskedDiffusionModel(nn.Module):
    def __init__(self, wyk_vocab: int, sg_vocab: int):
        super().__init__()
        d = D_MODEL
        self.wyk_vocab = wyk_vocab        # WYK_SPECIAL + W_RAW
        self.sg_vocab = sg_vocab          # N_SG + 1 (id 0 = MASK)
        self.tok_embed = nn.Embedding(VOCAB_SIZE, d)
        self.lat_axis_embed = nn.Embedding(N_LATTICE_TOK, d)
        self.coord_axis_embed = nn.Embedding(3, d)
        self.z_embed = nn.Embedding(MAX_Z + 1, d)
        # === per-atom positional embedding ===
        # Breaks atom-permutation equivariance: each ordered site slot gets a
        # learnable vector. Sites are presented in the canonical EN order
        # (train __getitem__ + sampler), so slot index carries a reproducible,
        # composition-derivable positional signal. Padded slots (i>=N) are
        # masked out by key_padding_mask, so their pos rows are harmless.
        self.elem_block_embed = nn.Embedding(MAX_N, d)   # rank of the atom's element block (EN order)
        self.within_elem_embed = nn.Embedding(MAX_N, d)  # rank within the element block
        self.sg_embed = nn.Embedding(sg_vocab, d)
        self.wyk_embed = nn.Embedding(wyk_vocab, d)
        self.time_embed = SinusoidalTimeEmbed(d)
        self.time_proj = nn.Linear(d, d)

        # per-atom bundle now 5 slots: 3 coord + Z + Wyckoff -> d
        self.site_mix = nn.Linear(d * 3 + d + d, d)

        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(d)

        self.head_lat = nn.Linear(d, K_BINS)
        self.head_coord = nn.Linear(d, 3 * K_BINS)
        self.head_sg = nn.Linear(d, sg_vocab)
        self.head_wyk = nn.Linear(d, wyk_vocab)
        # exp#1: continuous sub-bin offset regression heads (bin-width units).
        # Output bounded to [-0.5,0.5] via 0.5*tanh so it stays inside one bin.
        self.head_lat_off = nn.Linear(d, 1)
        self.head_coord_off = nn.Linear(d, 3)

    def forward(self, lat_tok, coord_tok, Z, sg_tok, wyk_tok, site_mask, t):
        B, N, _ = coord_tok.shape
        d = D_MODEL
        dev = lat_tok.device

        t_emb = self.time_proj(F.silu(self.time_embed(t)))

        # --- SG position (type id 1) ---
        sg_e = self.sg_embed(sg_tok)                                   # (B, d)
        sg_seq = sg_e.unsqueeze(1) + t_emb[:, None]  # (B, 1, d)

        # --- lattice positions (type id 0) ---
        lat_e = self.tok_embed(lat_tok)
        lat_axis = self.lat_axis_embed.weight[None].expand(B, -1, -1)
        lat_seq = lat_e + lat_axis + t_emb[:, None]

        # --- atom positions (type id 2): bundle coord + Z + Wyckoff ---
        coord_e = self.tok_embed(coord_tok)
        ax_e = self.coord_axis_embed.weight[None, None]
        coord_e = coord_e + ax_e
        coord_flat = coord_e.reshape(B, N, 3 * d)
        Z_e = self.z_embed(Z)
        wyk_e = self.wyk_embed(wyk_tok)
        site = self.site_mix(torch.cat([coord_flat, Z_e, wyk_e], dim=-1))
        site = site + t_emb[:, None]
        # factored positional: element-block rank k + within-element rank r, from Z (canonical order,
        # same-element atoms contiguous). Padded slots (~site_mask) -> 0 (masked out in attention).
        new_block = torch.ones(B, N, dtype=torch.long, device=dev)
        new_block[:, 1:] = (Z[:, 1:] != Z[:, :-1]).long()      # 1 where a new element block starts
        k = (torch.cumsum(new_block, dim=1) - 1)                # block rank, 0-based
        idx = torch.arange(N, device=dev).unsqueeze(0).expand(B, N)
        block_start = torch.cummax(torch.where(new_block.bool(), idx, torch.zeros_like(idx)), dim=1).values
        r = idx - block_start                                   # within-block rank
        k = (k * site_mask.long()).clamp(0, MAX_N - 1)
        r = (r * site_mask.long()).clamp(0, MAX_N - 1)
        site = site + self.elem_block_embed(k) + self.within_elem_embed(r)

        seq = torch.cat([sg_seq, lat_seq, site], dim=1)

        pad_sg = torch.zeros(B, 1, dtype=torch.bool, device=dev)
        pad_lat = torch.zeros(B, N_LATTICE_TOK, dtype=torch.bool, device=dev)
        pad_site = ~site_mask
        kpm = torch.cat([pad_sg, pad_lat, pad_site], dim=1)

        for blk in self.blocks:
            seq = blk(seq, key_padding_mask=kpm)
        seq = self.ln_f(seq)

        sg_out = seq[:, 0]                                  # (B, d)
        lat_out = seq[:, 1:1 + N_LATTICE_TOK]
        site_out = seq[:, 1 + N_LATTICE_TOK:]

        sg_logits = self.head_sg(sg_out)                    # (B, sg_vocab)
        lat_logits = self.head_lat(lat_out)
        coord_logits = self.head_coord(site_out).reshape(B, N, 3, K_BINS)
        wyk_logits = self.head_wyk(site_out)                # (B, N, wyk_vocab)
        # exp#1: bounded sub-bin offsets (bin-width units, [-0.5,0.5])
        lat_off = 0.5 * torch.tanh(self.head_lat_off(lat_out).squeeze(-1))   # (B,6)
        coord_off = 0.5 * torch.tanh(self.head_coord_off(site_out))          # (B,N,3)
        return lat_logits, coord_logits, sg_logits, wyk_logits, lat_off, coord_off


# ----------------------------- training ----------------------------------- #
def apply_absorbing_mask(tokens, t, valid_mask, mask_id):
    B = t.shape[0]
    view_shape = (B,) + (1,) * (tokens.ndim - 1)
    t_b = t.view(view_shape)
    rand = torch.rand_like(tokens, dtype=torch.float)
    mask = (rand < t_b) & valid_mask
    masked_tokens = torch.where(mask, torch.full_like(tokens, mask_id), tokens)
    return masked_tokens, mask


def train_one_step(model, batch, device):
    lat_tok = batch["lat_tok"].to(device, non_blocking=True)
    coord_tok = batch["coord_tok"].to(device, non_blocking=True)
    Z = batch["Z"].to(device, non_blocking=True)
    sg_tok = batch["sg_tok"].to(device, non_blocking=True)
    wyk_tok = batch["wyk_tok"].to(device, non_blocking=True)
    site_mask = batch["site_mask"].to(device, non_blocking=True)
    lat_offset_tgt = batch["lat_offset"].to(device, non_blocking=True)      # (B,6) exp#1
    coord_offset_tgt = batch["coord_offset"].to(device, non_blocking=True)  # (B,N,3) exp#1
    B, N, _ = coord_tok.shape

    t = torch.rand(B, device=device).clamp(1e-3, 1.0 - 1e-3)

    lat_valid = torch.ones_like(lat_tok, dtype=torch.bool)
    lat_in, lat_mm = apply_absorbing_mask(lat_tok, t, lat_valid, MASK_ID)

    coord_valid = site_mask[:, :, None].expand(B, N, 3)
    coord_in, coord_mm = apply_absorbing_mask(coord_tok, t, coord_valid, MASK_ID)

    sg_valid = torch.ones_like(sg_tok, dtype=torch.bool)
    sg_in, sg_mm = apply_absorbing_mask(sg_tok, t, sg_valid, SG_MASK)

    wyk_in, wyk_mm = apply_absorbing_mask(wyk_tok, t, site_mask, WYK_MASK)

    lat_logits, coord_logits, sg_logits, wyk_logits, lat_off, coord_off = model(
        lat_in, coord_in, Z, sg_in, wyk_in, site_mask, t
    )

    # lattice: ordinal Gaussian soft target, NON-CIRCULAR.
    # Mirrors the coord ordinal loss but with linear (non-wrapping) bin
    # distance: lattice params a,b,c,alpha,beta,gamma do NOT wrap.
    lat_logits_f = lat_logits.float().reshape(-1, K_BINS)
    lat_logp = F.log_softmax(lat_logits_f, dim=-1)
    lat_soft_tgt = lat_soft_targets_on(lat_logits_f.device)[lat_tok.reshape(-1)]
    loss_lat = -(lat_soft_tgt * lat_logp).sum(dim=-1).reshape(B, N_LATTICE_TOK)
    loss_lat = (loss_lat * lat_mm.float()).sum() / lat_mm.float().sum().clamp_min(1.0)

    # coord: ordinal Gaussian soft target
    coord_logits_f = coord_logits.float().reshape(-1, K_BINS)
    coord_logp = F.log_softmax(coord_logits_f, dim=-1)
    soft_tgt = coord_soft_targets_on(coord_logits_f.device)[coord_tok.reshape(-1)]
    loss_coord = -(soft_tgt * coord_logp).sum(dim=-1).reshape(B, N, 3)
    loss_coord = (loss_coord * coord_mm.float()).sum() / coord_mm.float().sum().clamp_min(1.0)

    # SG: flat CE over masked SG token
    loss_sg = F.cross_entropy(sg_logits.float(), sg_tok, reduction="none")
    loss_sg = (loss_sg * sg_mm.float()).sum() / sg_mm.float().sum().clamp_min(1.0)

    # Wyckoff: flat CE over masked, valid per-atom Wyckoff tokens
    loss_wyk = F.cross_entropy(
        wyk_logits.float().reshape(-1, model.wyk_vocab),
        wyk_tok.reshape(-1),
        reduction="none",
    ).reshape(B, N)
    loss_wyk = (loss_wyk * wyk_mm.float()).sum() / wyk_mm.float().sum().clamp_min(1.0)

    # exp#1: sub-bin offset regression (SmoothL1), supervised at UNMASKED valid
    # positions (input token == true bin), matching sampling-time offset reads.
    lat_off_sup = (lat_valid & ~lat_mm).float()                       # (B,6)
    loss_lat_off = F.smooth_l1_loss(lat_off.float(), lat_offset_tgt,
                                    beta=OFF_SMOOTH_BETA, reduction="none")
    loss_lat_off = (loss_lat_off * lat_off_sup).sum() / lat_off_sup.sum().clamp_min(1.0)

    coord_off_sup = (coord_valid & ~coord_mm).float()                 # (B,N,3)
    loss_coord_off = F.smooth_l1_loss(coord_off.float(), coord_offset_tgt,
                                      beta=OFF_SMOOTH_BETA, reduction="none")
    loss_coord_off = (loss_coord_off * coord_off_sup).sum() / coord_off_sup.sum().clamp_min(1.0)

    loss = (LOSS_W_LAT * loss_lat + LOSS_W_COORD * loss_coord
            + LOSS_W_SG * loss_sg + LOSS_W_WYK * loss_wyk
            + LOSS_W_LAT_OFF * loss_lat_off + LOSS_W_COORD_OFF * loss_coord_off)
    return (loss, loss_lat.detach(), loss_coord.detach(), loss_sg.detach(),
            loss_wyk.detach(), loss_lat_off.detach(), loss_coord_off.detach())


# ----------------------------- sampling ----------------------------------- #
@torch.no_grad()
def sample_batch(model, Z_pad, site_mask, device, steps=SAMPLE_STEPS,
                 temperature=SAMPLE_TEMPERATURE, gumbel_tau0=SAMPLE_GUMBEL_TAU0,
                 sampler_stats: dict | None = None):
    """
    MaskGIT iterative unmasking with Gumbel-perturbed confidence selection.
    Now jointly unmasks SG + per-atom Wyckoff streams alongside lattice and
    coord bins (all start fully masked). SG/Wyckoff are GENERATED, not given.
    """
    B, N = Z_pad.shape
    wyk_vocab = model.wyk_vocab

    lat_tok = torch.full((B, N_LATTICE_TOK), MASK_ID, dtype=torch.long, device=device)
    coord_tok = torch.full((B, N, 3), MASK_ID, dtype=torch.long, device=device)
    sg_tok = torch.full((B,), SG_MASK, dtype=torch.long, device=device)
    wyk_tok = torch.full((B, N), WYK_MASK, dtype=torch.long, device=device)

    lat_valid = torch.ones_like(lat_tok, dtype=torch.bool)
    coord_valid = site_mask[:, :, None].expand(B, N, 3)
    sg_valid = torch.ones_like(sg_tok, dtype=torch.bool)
    wyk_valid = site_mask

    total_pos = (lat_valid.float().sum(dim=-1)
                 + coord_valid.reshape(B, -1).float().sum(dim=-1)
                 + sg_valid.float()
                 + wyk_valid.float().sum(dim=-1))

    eps = 1e-9
    for step in range(steps):
        t_now = 1.0 - step / steps
        t_next = 1.0 - (step + 1) / steps
        gamma_next = math.cos(0.5 * math.pi * (1.0 - t_next))
        t_tensor = torch.full((B,), t_now, device=device)

        lat_logits, coord_logits, sg_logits, wyk_logits, _, _ = model(
            lat_tok, coord_tok, Z_pad, sg_tok, wyk_tok, site_mask, t_tensor
        )

        lat_logits = lat_logits.float() / temperature
        coord_logits = coord_logits.float() / temperature
        sg_logits = sg_logits.float() / temperature
        wyk_logits = wyk_logits.float() / temperature

        # forbid re-emitting a [MASK] id
        sg_logits[:, SG_MASK] = float("-inf")
        wyk_logits[:, :, WYK_MASK] = float("-inf")

        lat_probs = F.softmax(lat_logits, dim=-1)
        coord_probs = F.softmax(coord_logits, dim=-1)
        sg_probs = F.softmax(sg_logits, dim=-1)
        wyk_probs = F.softmax(wyk_logits, dim=-1)

        lat_samples = torch.multinomial(lat_probs.reshape(-1, K_BINS), 1).reshape(B, N_LATTICE_TOK)
        coord_samples = torch.multinomial(coord_probs.reshape(-1, K_BINS), 1).reshape(B, N, 3)
        sg_samples = torch.multinomial(sg_probs, 1).reshape(B)
        wyk_samples = torch.multinomial(wyk_probs.reshape(-1, wyk_vocab), 1).reshape(B, N)

        lat_logp = lat_probs.gather(-1, lat_samples.unsqueeze(-1)).squeeze(-1).clamp_min(eps).log()
        coord_logp = coord_probs.gather(-1, coord_samples.unsqueeze(-1)).squeeze(-1).clamp_min(eps).log()
        sg_logp = sg_probs.gather(-1, sg_samples.unsqueeze(-1)).squeeze(-1).clamp_min(eps).log()
        wyk_logp = wyk_probs.gather(-1, wyk_samples.unsqueeze(-1)).squeeze(-1).clamp_min(eps).log()

        tau = gumbel_tau0 * max(1.0 - step / steps, 0.0) ** 3
        if tau > 0:
            def perturb(lp):
                u = torch.rand_like(lp).clamp_min(eps)
                return lp + tau * (-torch.log(-torch.log(u)))
            lat_conf = perturb(lat_logp)
            coord_conf = perturb(coord_logp)
            sg_conf = perturb(sg_logp)
            wyk_conf = perturb(wyk_logp)
        else:
            lat_conf, coord_conf, sg_conf, wyk_conf = lat_logp, coord_logp, sg_logp, wyk_logp

        lat_mask_now = (lat_tok == MASK_ID) & lat_valid
        coord_mask_now = (coord_tok == MASK_ID) & coord_valid
        sg_mask_now = (sg_tok == SG_MASK) & sg_valid
        wyk_mask_now = (wyk_tok == WYK_MASK) & wyk_valid

        prop_lat = torch.where(lat_mask_now, lat_samples, lat_tok)
        prop_coord = torch.where(coord_mask_now, coord_samples, coord_tok)
        prop_sg = torch.where(sg_mask_now, sg_samples, sg_tok)
        prop_wyk = torch.where(wyk_mask_now, wyk_samples, wyk_tok)

        NEG_INF = torch.tensor(float("-inf"), device=device)
        POS_INF = torch.tensor(float("inf"), device=device)
        lat_score = torch.where(lat_mask_now, lat_conf, POS_INF.expand_as(lat_conf))
        coord_score = torch.where(coord_mask_now, coord_conf, POS_INF.expand_as(coord_conf))
        coord_score = torch.where(coord_valid, coord_score, POS_INF.expand_as(coord_score))
        sg_score = torch.where(sg_mask_now, sg_conf, POS_INF.expand_as(sg_conf))
        wyk_score = torch.where(wyk_mask_now, wyk_conf, POS_INF.expand_as(wyk_conf))
        wyk_score = torch.where(wyk_valid, wyk_score, POS_INF.expand_as(wyk_score))

        n_keep_masked = (gamma_next * total_pos).long().clamp(min=0)
        if step == steps - 1:
            n_keep_masked = torch.zeros_like(n_keep_masked)

        flat_score = torch.cat(
            [sg_score.unsqueeze(1), lat_score, coord_score.reshape(B, -1), wyk_score],
            dim=1,
        )
        sorted_scores, _ = flat_score.sort(dim=1)
        idx_clamp = n_keep_masked.clamp(max=flat_score.shape[1] - 1)
        threshold = sorted_scores.gather(1, idx_clamp.unsqueeze(1)).squeeze(1)
        threshold = torch.where(n_keep_masked > 0, threshold, NEG_INF.expand_as(threshold))

        keep_masked_flat = flat_score < threshold.unsqueeze(1)
        off = 0
        keep_sg = keep_masked_flat[:, off]; off += 1
        keep_lat_mask = keep_masked_flat[:, off:off + N_LATTICE_TOK]; off += N_LATTICE_TOK
        keep_coord_mask = keep_masked_flat[:, off:off + 3 * N].reshape(B, N, 3); off += 3 * N
        keep_wyk_mask = keep_masked_flat[:, off:off + N]

        sg_tok = torch.where(keep_sg, torch.full_like(prop_sg, SG_MASK), prop_sg)
        lat_tok = torch.where(keep_lat_mask, torch.full_like(prop_lat, MASK_ID), prop_lat)
        coord_tok = torch.where(keep_coord_mask, torch.full_like(prop_coord, MASK_ID), prop_coord)
        wyk_tok = torch.where(keep_wyk_mask, torch.full_like(prop_wyk, WYK_MASK), prop_wyk)

    residual_lat = (lat_tok == MASK_ID) & lat_valid
    residual_coord = (coord_tok == MASK_ID) & coord_valid
    residual_sg = (sg_tok == SG_MASK) & sg_valid
    residual_wyk = (wyk_tok == WYK_MASK) & wyk_valid
    n_residual = int(residual_lat.sum().item() + residual_coord.sum().item()
                     + residual_sg.sum().item() + residual_wyk.sum().item())
    if sampler_stats is not None:
        sampler_stats["residual_mask_count"] = sampler_stats.get("residual_mask_count", 0) + n_residual
        sampler_stats["calls"] = sampler_stats.get("calls", 0) + 1

    # exp#1: ALWAYS run a final t=0 forward over the committed sequence. It fills
    # any residual MASK tokens (argmax) AND reads the sub-bin offset heads for
    # every position -- offsets are only meaningful once bins are committed, so
    # this is the one forward whose offsets we keep.
    t_tensor = torch.zeros(B, device=device)
    lat_logits, coord_logits, sg_logits, wyk_logits, lat_off, coord_off = model(
        lat_tok, coord_tok, Z_pad, sg_tok, wyk_tok, site_mask, t_tensor
    )
    sg_logits = sg_logits.float(); sg_logits[:, SG_MASK] = float("-inf")
    wyk_logits = wyk_logits.float(); wyk_logits[:, :, WYK_MASK] = float("-inf")
    lat_tok = torch.where(lat_tok == MASK_ID, lat_logits.float().argmax(-1), lat_tok)
    coord_tok = torch.where(coord_tok == MASK_ID, coord_logits.float().argmax(-1), coord_tok)
    sg_tok = torch.where(sg_tok == SG_MASK, sg_logits.argmax(-1), sg_tok)
    wyk_tok = torch.where(wyk_tok == WYK_MASK, wyk_logits.argmax(-1), wyk_tok)

    coord_tok = torch.where(coord_tok == MASK_ID, torch.zeros_like(coord_tok), coord_tok)

    # SG/Wyckoff are generated then discarded for CIF decode (P1 output);
    # the decode path consumes lat_tok + coord_tok (+ committed sub-bin offsets).
    return lat_tok, coord_tok, lat_off.float(), coord_off.float()


# ------------------------ CIF writing ------------------------------------- #
ELEMENT_SYMBOLS = [
    "X","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
]


def write_cif(path: Path, params: np.ndarray, frac_coords: np.ndarray, atomic_numbers: np.ndarray):
    a, b, c, alpha, beta, gamma = params.tolist()
    lines = []
    lines.append(f"data_gen")
    lines.append("_symmetry_space_group_name_H-M   'P 1'")
    lines.append(f"_cell_length_a    {a:.6f}")
    lines.append(f"_cell_length_b    {b:.6f}")
    lines.append(f"_cell_length_c    {c:.6f}")
    lines.append(f"_cell_angle_alpha {alpha:.6f}")
    lines.append(f"_cell_angle_beta  {beta:.6f}")
    lines.append(f"_cell_angle_gamma {gamma:.6f}")
    lines.append("_symmetry_Int_Tables_number 1")
    lines.append("loop_")
    lines.append(" _symmetry_equiv_pos_site_id")
    lines.append(" _symmetry_equiv_pos_as_xyz")
    lines.append("  1  'x, y, z'")
    lines.append("loop_")
    lines.append(" _atom_site_type_symbol")
    lines.append(" _atom_site_label")
    lines.append(" _atom_site_fract_x")
    lines.append(" _atom_site_fract_y")
    lines.append(" _atom_site_fract_z")
    lines.append(" _atom_site_occupancy")
    counters = {}
    for i, Z in enumerate(atomic_numbers):
        sym = ELEMENT_SYMBOLS[int(Z)] if 0 < int(Z) < len(ELEMENT_SYMBOLS) else "X"
        counters[sym] = counters.get(sym, 0) + 1
        label = f"{sym}{counters[sym]}"
        fx, fy, fz = frac_coords[i]
        lines.append(f"  {sym}  {label}  {fx:.6f}  {fy:.6f}  {fz:.6f}  1.0")
    path.write_text("\n".join(lines) + "\n")


def jitter_coincident_sites(frac: np.ndarray, lat: np.ndarray, rng: np.random.Generator, min_dist=0.4):
    if frac.shape[0] < 2:
        return frac
    for _ in range(3):
        cart = frac @ lat
        ok = True
        for i in range(len(cart)):
            for j in range(i + 1, len(cart)):
                d = cart[i] - cart[j]
                if np.linalg.norm(d) < min_dist:
                    frac[j] = (frac[j] + (rng.random(3) - 0.5) * 0.1) % 1.0
                    ok = False
        if ok:
            break
    return frac


# ----------------------------- weight EMA --------------------------------- #
def update_ema(ema_state: dict, model: nn.Module, decay: float):
    """In-place EMA update of `ema_state` toward the live `model` weights.

    Floating-point tensors get the standard exponential moving average
    `ema = decay*ema + (1-decay)*param`; non-float buffers (e.g. int/long
    counters) are copied verbatim since an EMA of integer state is meaningless.
    """
    msd = model.state_dict()
    for k, v in ema_state.items():
        param = msd[k]
        if torch.is_floating_point(v):
            v.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        else:
            v.copy_(param)


# ------------------- reusable generation + eval --------------------------- #
def generate_cifs(model, records, out_dir, device, cur_steps, rng):
    """Sample one CIF per record into `out_dir/{idx:05d}.cif`, return n_written.

    Factored out of the original test-sampling block: groups records by atom
    count N, OOM-halves comps_per_call, runs sample_batch, dequantizes the
    lattice + fractional coords, jitters coincident sites, writes a CIF per
    record, then fills any missing index with a safe fallback CIF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_records = len(records)

    idx_by_N: dict[int, list[int]] = {}
    for i, r in enumerate(records):
        n = r["atomic_numbers"].shape[0]
        idx_by_N.setdefault(n, []).append(i)

    comps_per_call = SAMPLE_BATCH_TOTAL
    n_written = 0
    n_vol_fallback = 0
    n_cif_fallback = 0
    sampler_stats: dict = {}

    for N_size, idxs in sorted(idx_by_N.items(), reverse=True):
        start = 0
        while start < len(idxs):
            batch_idxs = idxs[start:start + comps_per_call]
            Bc = len(batch_idxs)
            try:
                Z_pad = torch.zeros(Bc, max(N_size, 1), dtype=torch.long, device=device)
                site_mask = torch.zeros(Bc, max(N_size, 1), dtype=torch.bool, device=device)
                for bi, vi in enumerate(batch_idxs):
                    z = records[vi]["atomic_numbers"].long()
                    # === canonical EN order ===
                    # Reorder sites by stable_argsort(EN[Z]) -- the SAME
                    # inter-element key the model saw in training. EN is a
                    # function of Z only, so this is composition-derivable; CIF
                    # output is set-invariant so reordering Z is safe.
                    if z.shape[0] > 1:
                        en_z = EN_TABLE_CPU[z.clamp(0, MAX_Z)]
                        order = torch.argsort(en_z, stable=True)
                        z = z[order]
                    z = z.to(device)
                    Z_pad[bi, :N_size] = z
                    site_mask[bi, :N_size] = True

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=(device.type == "cuda")):
                    lat_tok_b, coord_tok_b, lat_off_b, coord_off_b = sample_batch(
                        model, Z_pad, site_mask, device,
                        steps=cur_steps, sampler_stats=sampler_stats,
                    )
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                if comps_per_call > 1:
                    comps_per_call = max(1, comps_per_call // 2)
                    print(f"[sample] OOM, halving comps_per_call to {comps_per_call}", flush=True)
                    continue
                else:
                    print(f"[sample] OOM at comps=1 unrecoverable, skipping: {e}", flush=True)
                    start += max(1, len(batch_idxs))
                    continue

            lat_best_np = lat_tok_b.cpu().numpy()
            coord_best_np = coord_tok_b.cpu().numpy()
            lat_off_np = lat_off_b.cpu().numpy()       # exp#1 learned sub-bin offsets
            coord_off_np = coord_off_b.cpu().numpy()

            for bi, vi in enumerate(batch_idxs):
                params = dequantize_lat_params(lat_best_np[bi], offset=lat_off_np[bi])
                params[0:3] = np.clip(params[0:3], LEN_RANGE[0], LEN_RANGE[1])
                params[3:6] = np.clip(params[3:6], ANG_RANGE[0], ANG_RANGE[1])

                try:
                    L = params_to_lattice_matrix(params)
                except Exception:
                    L = params_to_lattice_matrix(np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0]))
                    params = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])

                frac = dequantize_frac(coord_best_np[bi, :N_size], offset=coord_off_np[bi, :N_size])
                frac = jitter_coincident_sites(frac, L, rng, min_dist=0.4)
                frac = frac % 1.0

                z_orig = records[vi]["atomic_numbers"].long()
                if z_orig.shape[0] > 1:
                    en_z = EN_TABLE_CPU[z_orig.clamp(0, MAX_Z)]
                    z_orig = z_orig[torch.argsort(en_z, stable=True)]
                Z_np = z_orig.numpy().astype(int)

                vol = abs(np.linalg.det(L))
                if vol < max(2.0 * N_size, 5.0):
                    s = (max(2.0 * N_size, 5.0) / vol) ** (1.0 / 3.0)
                    params[0:3] = np.clip(params[0:3] * s, LEN_RANGE[0], LEN_RANGE[1])
                    L = params_to_lattice_matrix(params)
                    n_vol_fallback += 1

                out_path = out_dir / f"{vi:05d}.cif"
                try:
                    write_cif(out_path, params, frac, Z_np)
                    n_written += 1
                except Exception:
                    try:
                        safe_params = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
                        safe_frac = (np.arange(N_size)[:, None] / max(N_size, 1)
                                     + np.zeros((N_size, 3))) * 0.9 + 0.05
                        write_cif(out_path, safe_params, safe_frac, Z_np)
                        n_written += 1
                        n_cif_fallback += 1
                    except Exception:
                        pass
            start += len(batch_idxs)

    # fallback CIF for any index whose CIF wasn't written
    for i in range(n_records):
        p = out_dir / f"{i:05d}.cif"
        if not p.exists():
            N = records[i]["atomic_numbers"].shape[0]
            Z_np = records[i]["atomic_numbers"].numpy().astype(int)
            safe_params = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
            safe_frac = rng.random((max(N, 1), 3))
            try:
                write_cif(p, safe_params, safe_frac, Z_np)
                n_written += 1
            except Exception:
                pass

    print(f"[sample] wrote {n_written}/{n_records} CIFs into {out_dir} "
          f"(steps={cur_steps}); vol_fallback={n_vol_fallback} "
          f"cif_fallback={n_cif_fallback}; sampler_calls={sampler_stats.get('calls', 0)} "
          f"residual_mask_total={sampler_stats.get('residual_mask_count', 0)}",
          flush=True)
    return n_written


def run_metre(samples_dir, ref_file, script_dir) -> float:
    """Compute the METRe match rate in [0, 1] for a directory of CIF samples.

    Calls ``evaluate.compute_metrics`` in-process and reads the value directly,
    so the metric never depends on the layout of evaluate.py's printed report.
    A failed measurement propagates as an exception (loud) rather than being
    swallowed into a sentinel the caller would mistake for "no improvement".
    """
    from evaluate import compute_metrics

    result = compute_metrics(
        samples_dir, str(ref_file),
        data_dir=str(Path(script_dir) / "data"),
        num_workers=0,
    )
    return result["match_rate"]


# ----------------------------- main --------------------------------------- #
def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mp_20",
                        choices=sorted(_DATASETS),
                        help="Dataset name (mp_20, mp_20_ps, mpts_52). Sets MAX_N "
                             "and the data/precompute file stems.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=3000)
    parser.add_argument("--val_every_epochs", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    # MAX_N was already resolved at import from --dataset (see _peek_dataset);
    # assert the parsed value agrees so a mismatch can never go unnoticed.
    assert args.dataset == DATASET, (
        f"dataset mismatch: import-time {DATASET!r} vs parsed {args.dataset!r}"
    )
    BATCH_SIZE = args.batch_size

    t_total_start = time.time()

    script_dir = Path(__file__).resolve().parent
    run_dir = script_dir / "runs" / args.run_name
    samples_dir = run_dir / "test_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load Wyckoff/SG precompute (fail fast if missing) ---
    wyk_index, wyk_cache, W_RAW = load_wyckoff_artifacts(script_dir, args.dataset)
    WYK_VOCAB = WYK_SPECIAL + W_RAW       # 0=MASK, 1=UNK, then W_RAW raw tokens
    SG_VOCAB = N_SG + 1                   # 0=MASK, 1..230 SG
    print(f"[init] device={device}, run_dir={run_dir}", flush=True)

    print(f"[init] wyckoff: W_RAW={W_RAW} WYK_VOCAB={WYK_VOCAB} SG_VOCAB={SG_VOCAB}", flush=True)
    print(f"[init] config: d={D_MODEL} depth={N_LAYERS} heads={N_HEADS} "
          f"batch={BATCH_SIZE} steps={SAMPLE_STEPS} "
          f"w_lat={LOSS_W_LAT} w_coord={LOSS_W_COORD} w_sg={LOSS_W_SG} w_wyk={LOSS_W_WYK}",
          flush=True)

    train_path = script_dir / "data" / f"{args.dataset}_train.pt"
    test_path = script_dir / "data" / f"{args.dataset}_test.pt"
    print(f"[data] loading train from {train_path}", flush=True)
    n_train = len(wyk_cache)
    normalizer = load_normalizer(script_dir, wyk_index)
    train_ds = CrystalDataset(
        str(train_path), wyk_cache, wyk_index,
        normalizer=normalizer,
        shift_aug_p=SHIFT_AUG_P, orbit_perm_p=ORBIT_PERM_P,
        coord_noise_p=COORD_NOISE_P,
        coord_noise_sigma_bin=COORD_NOISE_SIGMA_BIN,
    )
    print(f"[permsym] per-atom pos_embed ON; canonical EN sort + "
          f"intra/inter-orbit perm aug (p={ORBIT_PERM_P}); sample order = "
          f"stable_argsort(EN[Z])", flush=True)
    print(f"[coordnoise] sub-bin Gaussian coord-noise aug: "
          f"sigma={COORD_NOISE_SIGMA_BIN:.3f} bins "
          f"(={COORD_NOISE_SIGMA_BIN/K_BINS:.5f} frac), p={COORD_NOISE_P}, "
          f"applied AFTER shift+orbit-perm, BEFORE quantize", flush=True)
    if not normalizer.get("enabled"):
        print("[aug] running WITHOUT online normalizer augmentation (fallback).", flush=True)
    else:
        print("[aug] online Euclidean-normalizer translation aug ENABLED "
              f"(p={SHIFT_AUG_P}); replaces offline shift_aug.pt", flush=True)

    def make_loader(bs):
        return DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=2, collate_fn=collate, pin_memory=True,
                          persistent_workers=True, drop_last=True)

    current_bs = BATCH_SIZE
    train_dl = make_loader(current_bs)

    model = MaskedDiffusionModel(WYK_VOCAB, SG_VOCAB).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M", flush=True)

    # --- weight EMA: clone of the model state, kept on the model's device ---
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print(f"[ema] initialized EMA state ({len(ema_state)} tensors) "
          f"decay={args.ema_decay}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                            betas=(0.9, 0.95))

    compiled = False
    compile_t0 = time.time()
    try:
        model_run = torch.compile(model, mode="reduce-overhead", dynamic=True)
        with torch.no_grad():
            B0 = 2
            lat_tok_w = torch.zeros(B0, N_LATTICE_TOK, dtype=torch.long, device=device)
            coord_tok_w = torch.zeros(B0, 4, 3, dtype=torch.long, device=device)
            Z_w = torch.ones(B0, 4, dtype=torch.long, device=device)
            sg_w = torch.zeros(B0, dtype=torch.long, device=device)
            wyk_w = torch.zeros(B0, 4, dtype=torch.long, device=device)
            ep_w = torch.zeros(B0, 4, dtype=torch.long, device=device)
            sm_w = torch.ones(B0, 4, dtype=torch.bool, device=device)
            t_w = torch.full((B0,), 0.5, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model_run(lat_tok_w, coord_tok_w, Z_w, sg_w, wyk_w, ep_w, sm_w, t_w)
        compiled = True
        print(f"[compile] torch.compile ok in {time.time()-compile_t0:.1f}s", flush=True)
    except Exception as e:
        model_run = model
        print(f"[compile] DISABLED (eager fallback): {type(e).__name__}: {e}", flush=True)

    # --- val records (loaded once before the loop) ---
    val_path = script_dir / "data" / f"{args.dataset}_val.pt"
    print(f"[data] loading val from {val_path}", flush=True)
    val_records = torch.load(str(val_path), weights_only=False)
    print(f"[data] n_val={len(val_records)}", flush=True)

    # --- epoch / step accounting and cosine schedule ---
    steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
    total_steps = max(1, args.max_epochs * steps_per_epoch)

    def swap_to_ema():
        """Back up live weights to CPU, load EMA weights, set eval mode."""
        backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state)
        model.eval()
        return backup

    def restore_live(backup):
        """Move backed-up live weights back to device and resume training."""
        model.load_state_dict({k: v.to(device) for k, v in backup.items()})
        model.train()

    def run_validation(epoch_idx, gstep):
        rng_v = np.random.default_rng(0)
        backup = swap_to_ema()
        try:
            generate_cifs(model, val_records, run_dir / "val_samples",
                          device, SAMPLE_STEPS, rng_v)
            vm = run_metre(run_dir / "val_samples", f"{args.dataset}_val.pt", script_dir)
        finally:
            restore_live(backup)
        return vm

    model.train()
    t_train_start = time.time()
    global_step = 0
    log_every = 50
    loss_ema = None

    best_val = float("-inf")
    best_epoch = -1
    no_improve = 0
    early_stopped = False

    print(f"[train] epoch-based: max_epochs={args.max_epochs} "
          f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
          f"val_every={args.val_every_epochs} patience={args.early_stop_patience} "
          f"compiled={compiled}", flush=True)

    for epoch in range(args.max_epochs):
        for batch in train_dl:
            frac = min(1.0, global_step / total_steps)
            lr_now = LR * (LR_FLOOR_FRAC + (1 - LR_FLOOR_FRAC) * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr_now

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                loss, ll, lc, lsg, lwyk, llo, lco = train_one_step(model_run, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            update_ema(ema_state, model, args.ema_decay)

            loss_v = loss.item()
            loss_ema = loss_v if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_v
            global_step += 1
            if global_step % log_every == 0:
                print(f"[train] step={global_step} epoch={epoch+1} "
                      f"loss={loss_v:.4f} ema={loss_ema:.4f} "
                      f"lat={ll.item():.4f} coord={lc.item():.4f} "
                      f"sg={lsg.item():.4f} wyk={lwyk.item():.4f} "
                      f"latoff={llo.item():.4f} coordoff={lco.item():.4f} "
                      f"lr={lr_now:.2e} bs={current_bs}",
                      flush=True)

        # ---------- periodic validation on the full val set (EMA weights) ----
        if (epoch + 1) % args.val_every_epochs == 0:
            val_metre = run_validation(epoch, global_step)
            improved = val_metre > best_val
            if improved:
                best_val = val_metre
                best_epoch = epoch + 1
                torch.save(ema_state, run_dir / "best.pt")
                no_improve = 0
            else:
                no_improve += 1
            print(f"[val] epoch={epoch+1} step={global_step} "
                  f"val_metre={val_metre:.4f} best={best_val:.4f}", flush=True)
            if improved:
                print(f"[val] *** new best -> saved best.pt", flush=True)
            if no_improve >= args.early_stop_patience:
                print(f"[val] early stop: no_improve={no_improve} >= "
                      f"patience={args.early_stop_patience}", flush=True)
                early_stopped = True
                break

    training_seconds = time.time() - t_train_start
    print(f"[train] done: steps={global_step} epochs={epoch+1} "
          f"early_stopped={early_stopped} training_seconds={training_seconds:.1f}",
          flush=True)

    # ---------- final test evaluation on the best EMA checkpoint ---------- #
    best_path = run_dir / "best.pt"
    print(f"[sample] loading test from {test_path}", flush=True)
    test_records = torch.load(str(test_path), weights_only=False)
    n_test = len(test_records)
    print(f"[sample] n_test={n_test}", flush=True)

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    test_metre = float("nan")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        generate_cifs(model, test_records, samples_dir, device, SAMPLE_STEPS, rng)
        test_metre = run_metre(samples_dir, f"{args.dataset}_test.pt", script_dir)
    else:
        print("[sample] no best.pt found (no validation improvement recorded); "
              "skipping test evaluation.", flush=True)

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    total_seconds = time.time() - t_total_start

    print("---", flush=True)
    print(f"best_val_metre:   {best_val if best_val != float('-inf') else float('nan')}", flush=True)
    print(f"best_epoch:       {best_epoch}", flush=True)
    print(f"test_metre:       {test_metre}", flush=True)
    print(f"training_seconds: {training_seconds:.2f}", flush=True)
    print(f"total_seconds:    {total_seconds:.2f}", flush=True)
    print(f"peak_vram_mb:     {peak_vram_mb:.2f}", flush=True)


if __name__ == "__main__":
    main()
