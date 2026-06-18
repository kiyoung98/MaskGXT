"""Unified sampling entry point for MaskGXT.

Generates one CIF per reference record (index-aligned: ``{idx:05d}.cif``), so the
output directory can be scored directly with ``evaluate.py``.

Two INDEPENDENT decode controls:

  --greedy        MAP / argmax token selection (zeroed Gumbel confidence noise).
                  Scope depends on --sg_stratify: without it, EVERY generation is
                  greedy; with it, only each SG group's anchor is greedy and the
                  rest stay stochastic (for polymorph diversity).

  --sg_stratify   SG-stratified sampling. Per composition with multiplicity K_C,
                  the K_C generations are assigned distinct top space groups from
                  the model's first-step SG posterior (without-replacement above a
                  posterior-mass floor), so they cover distinct polymorphs rather
                  than collapsing onto the dominant one.

The two flags map onto the paper settings:

  match rate       --greedy                 (i.i.d. SG, all generations argmax)
  METRe (i.i.d.)   (no flags)               (i.i.d. SG, all stochastic)
  METRe (full)     --greedy --sg_stratify   (SG-stratified, anchor argmax)

The model, tokenization, and dequantization are imported from ``train.py`` (one
source of truth, so the sampler always matches the trained checkpoint). The
SG-stratification / anchor-greedy decode logic lives in this file because it is
sampling-only logic absent from training. Decode config (canonical minimal
core): ``SG_POST_FLOOR=0.02``, lattice repulsion off, non-greedy temperature 1.0.

Usage:
    # single-best match rate (i.i.d. SG, all-argmax)
    python sample.py --dataset mp_20 --greedy --ckpt runs/<run>/best.pt
    # polymorph-aware METRe (SG-stratified + greedy anchor)
    python sample.py --dataset mp_20 --greedy --sg_stratify --ckpt runs/<run>/best.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter as _Counter
from functools import reduce as _reduce
from math import gcd as _gcd
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Select the dataset BEFORE importing train, so train's import-time MAX_N
# resolves correctly (train.py reads MASKGXT_DATASET / --dataset at import).
_DS = None
for _i, _a in enumerate(sys.argv):
    if _a == "--dataset" and _i + 1 < len(sys.argv):
        _DS = sys.argv[_i + 1]
    elif _a.startswith("--dataset="):
        _DS = _a.split("=", 1)[1]
if _DS is not None:
    os.environ["MASKGXT_DATASET"] = _DS

import train as T

# --- names the grafted polymorph decoder references as module globals ---
# (Single source of truth: everything below is the trained model's own
# tokenization / dequantization / constants, imported from train.py.)
K_BINS = T.K_BINS
MASK_ID = T.MASK_ID
MAX_Z = T.MAX_Z
N_SG = T.N_SG
N_LATTICE_TOK = T.N_LATTICE_TOK
LEN_RANGE = T.LEN_RANGE
ANG_RANGE = T.ANG_RANGE
SG_MASK = T.SG_MASK
WYK_MASK = T.WYK_MASK
WYK_SPECIAL = T.WYK_SPECIAL
EN_TABLE_CPU = T.EN_TABLE_CPU
SAMPLE_STEPS = T.SAMPLE_STEPS
SAMPLE_BATCH_TOTAL = T.SAMPLE_BATCH_TOTAL
SAMPLE_TEMPERATURE = T.SAMPLE_TEMPERATURE
SAMPLE_GUMBEL_TAU0 = T.SAMPLE_GUMBEL_TAU0
MaskedDiffusionModel = T.MaskedDiffusionModel
dequantize_frac = T.dequantize_frac
dequantize_lat_params = T.dequantize_lat_params
params_to_lattice_matrix = T.params_to_lattice_matrix
write_cif = T.write_cif
jitter_coincident_sites = T.jitter_coincident_sites


# ============================================================================
# Sampling-only decode logic (SG-stratification + anchor-greedy override),
# grafted from the sampling-only extraction; operates on train.py's model.
# The --greedy and --sg_stratify flags select which parts engage.
# ============================================================================

# Posterior mass floor: an SG must hold at least this much first-step mass to
# be eligible as a clamp target for a non-anchor chain. Below this we treat it
# as unsupported and fall back to i.i.d. (guards precision against OOD clamps).
# (KEPT, 783920c core — the near-optimal ~2039-row SG-clamp reach.)
SG_POST_FLOOR = 0.02

# Sentinel effective-SG id for FREE (sg_clamp<0) chains: all free chains in a
# composition group share one bucket (they will i.i.d.-sample SG and tend to
# collapse onto the dominant SG -> treat them as one colliding remnant).
_FREE_SG = 0



def assign_group_space_groups(group_ids, sg_post, sg_floor=SG_POST_FLOOR):
    """Greedy SG-stratified WITHOUT-REPLACEMENT assignment per composition group.

    Args:
      group_ids: (B,) long/int tensor or list -- records sharing a value are
                 the same reduced-formula composition (one pooled bucket).
      sg_post:   (B, sg_vocab) float tensor on CPU -- each record's first-step
                 (t=1, fully-masked) SG posterior with SG_MASK already zeroed.
      sg_floor:  minimum posterior mass for an SG to be a clamp target.

    Returns:
      sg_clamp: (B,) long tensor. sg_clamp[b] in [1, N_SG] => pin that SG for
                record b; sg_clamp[b] == -1 => leave SG free (i.i.d. path).

    Policy (weakly dominates i.i.d. on pooled coverage):
      - Singleton groups (K_C == 1): sg_clamp = -1 (untouched i.i.d. path).
      - Multi groups: order members by descending top-1 confidence. The most
        confident member is the ANCHOR (chain 0) and is pinned to its MAP SG
        (its argmax) -- the dominant mode is never lost. Each subsequent
        member is pinned to the highest-posterior SG (by ITS OWN posterior)
        not yet used in the group AND with mass >= sg_floor; if none qualifies
        the member falls back to free i.i.d. sampling (sg_clamp = -1).
    """
    if not torch.is_tensor(group_ids):
        group_ids = torch.as_tensor(group_ids)
    group_ids = group_ids.cpu()
    B = sg_post.shape[0]
    sg_clamp = torch.full((B,), -1, dtype=torch.long)

    # bucket record indices by group id
    buckets: dict[int, list[int]] = {}
    for b in range(B):
        buckets.setdefault(int(group_ids[b].item()), []).append(b)

    top1 = sg_post.max(dim=-1).values  # (B,) confidence of each record's MAP SG

    for gid, members in buckets.items():
        if len(members) <= 1:
            continue  # singleton -> untouched i.i.d. path (sg_clamp stays -1)
        # order members by descending MAP confidence: most confident = anchor
        members_sorted = sorted(members, key=lambda b: float(top1[b]), reverse=True)
        used: set[int] = set()
        for rank, b in enumerate(members_sorted):
            post_b = sg_post[b]
            if rank == 0:
                # ANCHOR: pin to MAP SG (dominant mode never lost)
                sg = int(post_b.argmax().item())
                sg_clamp[b] = sg
                used.add(sg)
                continue
            # subsequent chains: highest-posterior UNUSED SG above the floor
            order = torch.argsort(post_b, descending=True)
            chosen = -1
            for sg_id in order.tolist():
                if sg_id == SG_MASK:
                    continue
                if sg_id in used:
                    continue
                if float(post_b[sg_id]) < sg_floor:
                    break  # remaining are below floor (sorted) -> stop
                chosen = sg_id
                break
            if chosen >= 1:
                sg_clamp[b] = chosen
                used.add(chosen)
            # else: leave -1 -> fall back to free i.i.d. SG sampling
    return sg_clamp


@torch.no_grad()
def probe_sg_posterior(model, Z_pad, site_mask, device,
                       temperature=SAMPLE_TEMPERATURE):
    """One forward at t=1 (fully masked) -> per-record SG posterior (CPU).

    Conditioned only on (Z, site_mask) -- composition-derivable;
    reads NO val coords/lattice. Returns (B, sg_vocab) float on CPU with the
    SG_MASK column zeroed. This is the model's first-step SG belief, the
    handle for cross-sample stratification.
    """
    B, N = Z_pad.shape
    lat_tok = torch.full((B, N_LATTICE_TOK), MASK_ID, dtype=torch.long, device=device)
    coord_tok = torch.full((B, N, 3), MASK_ID, dtype=torch.long, device=device)
    sg_tok = torch.full((B,), SG_MASK, dtype=torch.long, device=device)
    wyk_tok = torch.full((B, N), WYK_MASK, dtype=torch.long, device=device)
    t_tensor = torch.ones(B, device=device)
    _, _, sg_logits, _, _, _ = model(
        lat_tok, coord_tok, Z_pad, sg_tok, wyk_tok, site_mask, t_tensor
    )
    sg_logits = sg_logits.float() / temperature
    sg_logits[:, SG_MASK] = float("-inf")
    sg_post = F.softmax(sg_logits, dim=-1)
    return sg_post.cpu()


def _derive_group_anchors(Z_pad, site_mask, sg_clamp):
    """Flag the anchor row of each effective-SG group.

    Rows are grouped by (composition, effective SG), where:
      * composition = the (already EN-sorted) atomic-number vector over valid
        sites (rows with identical Z are the same pooled polymorph bucket).
      * effective SG = sg_clamp[b] if pinned (>=1), else the shared FREE
        sentinel _FREE_SG (all free/i.i.d. chains of a composition share one
        bucket).

    Returns is_anchor (B,) bool: True for exactly one row per group (the lowest
    index = the parent's MAP/dominant chain). This is the row decoded greedily
    when --greedy --sg_stratify; every other row in the group stays stochastic.
    Singletons and SG-distinct chains form groups of size 1 -> is_anchor=True.
    """
    B, N = Z_pad.shape
    z_cpu = Z_pad.detach().cpu()
    sm_cpu = site_mask.detach().cpu()
    sc_cpu = sg_clamp.detach().cpu() if sg_clamp is not None else None
    is_anchor = torch.zeros(B, dtype=torch.bool)
    seen: set[tuple] = set()
    for b in range(B):
        n = int(sm_cpu[b].sum().item())
        comp = tuple(int(x) for x in z_cpu[b, :n].tolist())
        if sc_cpu is not None and int(sc_cpu[b].item()) >= 1:
            eff_sg = int(sc_cpu[b].item())
        else:
            eff_sg = _FREE_SG
        key = (comp, eff_sg)
        if key not in seen:
            is_anchor[b] = True
            seen.add(key)
    return is_anchor


@torch.no_grad()
def sample_batch(model, Z_pad, site_mask, device, steps=SAMPLE_STEPS,
                 temperature=SAMPLE_TEMPERATURE, gumbel_tau0=SAMPLE_GUMBEL_TAU0,
                 sampler_stats: dict | None = None, sg_clamp=None, greedy=False):
    """
    MaskGIT iterative unmasking with Gumbel-perturbed confidence selection.
    Jointly unmasks SG + per-atom Wyckoff streams alongside lattice and coord
    bins (all start fully masked). SG/Wyckoff are GENERATED, not given.

    Two independent decode controls:

    * ``greedy`` -- MAP/argmax token selection (and zeroed Gumbel confidence
      noise). Its SCOPE depends on stratification: when SG-stratified
      (``sg_clamp`` given), only the per-group anchor rows are decoded greedily
      (the rest stay stochastic for polymorph diversity); when NOT stratified
      (``sg_clamp is None``), EVERY row is decoded greedily (the single-best
      match-rate setting). ``greedy=False`` keeps all rows stochastic.

    * ``sg_clamp`` (SG stratification) -- if given, records with
      sg_clamp[b] >= 1 have their SG token PINNED and removed from the unmask
      set; records with sg_clamp[b] < 0 (and all records when sg_clamp is None)
      follow the i.i.d. SG path.
    """
    B, N = Z_pad.shape
    wyk_vocab = model.wyk_vocab

    # --- per-row greedy (argmax) mask. SG-stratified runs decode only each
    # effective-SG group's anchor greedily (the rest stay stochastic for
    # polymorph diversity); unstratified greedy runs decode EVERY row greedily
    # (single-best match-rate setting). greedy=False -> no row is greedy. ---
    is_anchor = _derive_group_anchors(Z_pad, site_mask, sg_clamp).to(device)
    if not greedy:
        greedy_mask = torch.zeros(B, dtype=torch.bool, device=device)
    elif sg_clamp is None:
        greedy_mask = torch.ones(B, dtype=torch.bool, device=device)
    else:
        greedy_mask = is_anchor

    lat_tok = torch.full((B, N_LATTICE_TOK), MASK_ID, dtype=torch.long, device=device)
    coord_tok = torch.full((B, N, 3), MASK_ID, dtype=torch.long, device=device)
    sg_tok = torch.full((B,), SG_MASK, dtype=torch.long, device=device)
    wyk_tok = torch.full((B, N), WYK_MASK, dtype=torch.long, device=device)

    lat_valid = torch.ones_like(lat_tok, dtype=torch.bool)
    coord_valid = site_mask[:, :, None].expand(B, N, 3)
    sg_valid = torch.ones_like(sg_tok, dtype=torch.bool)
    wyk_valid = site_mask

    # --- SG clamp (parent stage 1): pin assigned space groups, remove from unmask ---
    if sg_clamp is not None:
        sg_clamp = sg_clamp.to(device)
        clamped = sg_clamp >= 1
        if clamped.any():
            sg_tok = torch.where(clamped, sg_clamp.clamp(min=SG_MASK), sg_tok)
            sg_valid = sg_valid & (~clamped)

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

        # === GREEDY / MAP token selection for greedy_mask rows ===
        # For greedy rows, OVERRIDE the stochastic multinomial draw with the
        # per-position ARGMAX (most-probable token) so the draw concentrates on
        # the model's MAP structure. Non-greedy rows keep their stochastic sample
        # -> diversity preserved. Scope of greedy_mask: all rows (unstratified
        # greedy, match-rate setting) or only effective-SG anchors (stratified
        # greedy, METRe setting). We override on the (B,..) sample tensors per
        # row; logp is then gathered on the chosen token below, so the
        # confidence/unmask schedule is consistent either way.
        if bool(greedy_mask.any()):
            a_l = greedy_mask[:, None].expand(B, N_LATTICE_TOK)
            a_c = greedy_mask[:, None, None].expand(B, N, 3)
            a_w = greedy_mask[:, None].expand(B, N)
            lat_samples = torch.where(a_l, lat_probs.argmax(dim=-1), lat_samples)
            coord_samples = torch.where(a_c, coord_probs.argmax(dim=-1), coord_samples)
            sg_samples = torch.where(greedy_mask, sg_probs.argmax(dim=-1), sg_samples)
            wyk_samples = torch.where(a_w, wyk_probs.argmax(dim=-1), wyk_samples)
        # ============================================================================

        lat_logp = lat_probs.gather(-1, lat_samples.unsqueeze(-1)).squeeze(-1).clamp_min(eps).log()
        coord_logp = coord_probs.gather(-1, coord_samples.unsqueeze(-1)).squeeze(-1).clamp_min(eps).log()
        sg_logp = sg_probs.gather(-1, sg_samples.unsqueeze(-1)).squeeze(-1).clamp_min(eps).log()
        wyk_logp = wyk_probs.gather(-1, wyk_samples.unsqueeze(-1)).squeeze(-1).clamp_min(eps).log()

        tau = gumbel_tau0 * max(1.0 - step / steps, 0.0) ** 3
        if tau > 0:
            # Gumbel-perturbed confidence ordering for the STOCHASTIC chains. For
            # GREEDY rows we ZERO the perturbation (per-row) so their unmask
            # ORDER follows pure model confidence, not injected noise -- consistent
            # with their argmax (MAP) token selection above. The gate broadcasts
            # over each stream's position axes.
            def perturb(lp):
                u = torch.rand_like(lp).clamp_min(eps)
                noise = tau * (-torch.log(-torch.log(u)))
                if bool(greedy_mask.any()):
                    keep = (~greedy_mask).float()
                    # reshape keep to broadcast over lp's trailing dims
                    while keep.dim() < lp.dim():
                        keep = keep.unsqueeze(-1)
                    noise = noise * keep
                return lp + noise
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

    if residual_lat.any() or residual_coord.any() or residual_sg.any() or residual_wyk.any():
        t_tensor = torch.zeros(B, device=device)
        lat_logits, coord_logits, sg_logits, wyk_logits, _, _ = model(
            lat_tok, coord_tok, Z_pad, sg_tok, wyk_tok, site_mask, t_tensor
        )
        sg_logits = sg_logits.float(); sg_logits[:, SG_MASK] = float("-inf")
        wyk_logits = wyk_logits.float(); wyk_logits[:, :, WYK_MASK] = float("-inf")
        lat_tok = torch.where(lat_tok == MASK_ID, lat_logits.float().argmax(-1), lat_tok)
        coord_tok = torch.where(coord_tok == MASK_ID, coord_logits.float().argmax(-1), coord_tok)
        sg_tok = torch.where(sg_tok == SG_MASK, sg_logits.argmax(-1), sg_tok)
        wyk_tok = torch.where(wyk_tok == WYK_MASK, wyk_logits.argmax(-1), wyk_tok)

    coord_tok = torch.where(coord_tok == MASK_ID, torch.zeros_like(coord_tok), coord_tok)

    # contoffset: one final t=0 forward (ALWAYS) reads the learned sub-bin
    # offsets at the committed tokens; these REPLACE random jitter at decode.
    t0 = torch.zeros(B, device=device)
    _, _, _, _, lat_off, coord_off = model(
        lat_tok, coord_tok, Z_pad, sg_tok, wyk_tok, site_mask, t0
    )

    # SG/Wyckoff are generated then discarded for CIF decode (P1 output);
    # the decode path consumes lat_tok + coord_tok + the predicted offsets.
    return lat_tok, coord_tok, lat_off.float(), coord_off.float()
# ==================== END DECODER (search space) ====================


def generate_cifs(model, records, out_dir, device, cur_steps, rng,
                  greedy=False, sg_stratify=False):
    """Sample one CIF per record into `out_dir/{idx:05d}.cif`, return n_written.

    Groups records by atom count N, OOM-halves comps_per_call, runs
    sample_batch, dequantizes the lattice + fractional coords, jitters
    coincident sites, writes a CIF per record, then fills any missing index
    with a safe fallback CIF.

    Two independent decode controls (see sample_batch):
      * ``greedy`` -- MAP/argmax token selection. Unstratified: every row;
        stratified: only the per-group anchor.
      * ``sg_stratify`` -- if True, run the SG-stratification PASS 1 below:
        probe each multi-bucket record's first-step SG posterior and greedily
        assign a DISTINCT clamp SG per record within each composition group
        (anchor=MAP, chains 1..K_C-1 = next-best unused SG above the floor, else
        i.i.d. fallback). If False, all records follow the i.i.d. SG path.

    CIFs are written to the ORIGINAL record idx, so idx-alignment to the
    reference split order is preserved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_records = len(records)

    # --- composition grouping (decoder substrate) ---
    group_ids = compute_group_ids(records)              # one int per record
    from collections import Counter as _Counter
    _gc = _Counter(group_ids)
    n_multi_records = sum(1 for g in group_ids if _gc[g] >= 2)

    idx_by_N: dict[int, list[int]] = {}
    for i, r in enumerate(records):
        n = r["atomic_numbers"].shape[0]
        idx_by_N.setdefault(n, []).append(i)

    comps_per_call = SAMPLE_BATCH_TOTAL
    n_written = 0
    n_vol_fallback = 0
    n_cif_fallback = 0
    sampler_stats: dict = {}

    # ---- build per-record EN-ordered model inputs once (reused for probe) ---
    def build_inputs(batch_idxs, N_size):
        Bc = len(batch_idxs)
        Z_pad = torch.zeros(Bc, max(N_size, 1), dtype=torch.long, device=device)
        site_mask = torch.zeros(Bc, max(N_size, 1), dtype=torch.bool, device=device)
        for bi, vi in enumerate(batch_idxs):
            z = records[vi]["atomic_numbers"].long()
            # === canonical EN order === (composition-derivable; CIF set-invariant)
            if z.shape[0] > 1:
                en_z = EN_TABLE_CPU[z.clamp(0, MAX_Z)]
                order = torch.argsort(en_z, stable=True)
                z = z[order]
            Z_pad[bi, :N_size] = z.to(device)
            site_mask[bi, :N_size] = True
        return Z_pad, site_mask

    # ---- PASS 1: probe first-step SG posteriors for records in multi-buckets --
    # Only records that share a composition group need coordination; probing
    # only them keeps the extra forward cost to ~28% of records, batched by N.
    # sg_clamp_per_idx defaults to -1 (i.i.d.) for everyone (incl. singletons).
    sg_clamp_per_idx = [-1] * n_records
    if sg_stratify and n_multi_records > 0:
        multi_idx_by_N: dict[int, list[int]] = {}
        for i in range(n_records):
            if _gc[group_ids[i]] >= 2:
                n = records[i]["atomic_numbers"].shape[0]
                multi_idx_by_N.setdefault(n, []).append(i)

        # collect posteriors per record index
        post_per_idx: dict[int, torch.Tensor] = {}
        probe_cpc = comps_per_call
        for N_size, idxs in sorted(multi_idx_by_N.items(), reverse=True):
            start = 0
            while start < len(idxs):
                batch_idxs = idxs[start:start + probe_cpc]
                try:
                    Z_pad, site_mask = build_inputs(batch_idxs, N_size)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                            enabled=(device.type == "cuda")):
                        sg_post = probe_sg_posterior(
                            model, Z_pad, site_mask, device
                        )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if probe_cpc > 1:
                        probe_cpc = max(1, probe_cpc // 2)
                        continue
                    else:
                        # cannot probe -> leave these as i.i.d. (-1)
                        start += max(1, len(batch_idxs))
                        continue
                for bi, vi in enumerate(batch_idxs):
                    post_per_idx[vi] = sg_post[bi]
                start += len(batch_idxs)

        # greedy WITHOUT-REPLACEMENT SG assignment, per composition group
        probed = sorted(post_per_idx.keys())
        if probed:
            gid_t = torch.tensor([group_ids[i] for i in probed], dtype=torch.long)
            post_t = torch.stack([post_per_idx[i] for i in probed], dim=0)
            sg_assign = assign_group_space_groups(gid_t, post_t)
            for j, vi in enumerate(probed):
                sg_clamp_per_idx[vi] = int(sg_assign[j].item())
        n_clamped = sum(1 for v in sg_clamp_per_idx if v >= 1)
        print(f"[coord] multi-bucket records={n_multi_records} probed={len(probed)} "
              f"SG-clamped={n_clamped} (rest fall back to i.i.d.); "
              f"SG-stratify + greedy anchor (minimal core)", flush=True)

    # ---- PASS 2: main sampling (batched by N), passing per-record SG clamp ---
    for N_size, idxs in sorted(idx_by_N.items(), reverse=True):
        start = 0
        while start < len(idxs):
            batch_idxs = idxs[start:start + comps_per_call]
            Bc = len(batch_idxs)
            try:
                Z_pad, site_mask = build_inputs(batch_idxs, N_size)
                # When not stratifying, pass sg_clamp=None so sample_batch takes
                # the unstratified path (i.i.d. SG; greedy scope = all rows).
                if sg_stratify:
                    sg_clamp_b = torch.tensor(
                        [sg_clamp_per_idx[vi] for vi in batch_idxs],
                        dtype=torch.long, device=device,
                    )
                else:
                    sg_clamp_b = None
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=(device.type == "cuda")):
                    lat_tok_b, coord_tok_b, lat_off_b, coord_off_b = sample_batch(
                        model, Z_pad, site_mask, device,
                        steps=cur_steps, sampler_stats=sampler_stats,
                        sg_clamp=sg_clamp_b, greedy=greedy,
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
            lat_off_np = lat_off_b.cpu().numpy()        # contoffset learned offsets
            coord_off_np = coord_off_b.cpu().numpy()

            # Plain shared-rng i.i.d. sub-bin jitter (KEPT, 783920c / frozen
            # dequant). The antithetic / van-der-Corput JITTER COUPLING (parent
            # 32b40b6) was confirmed seed-noise and is REMOVED this node; dequant
            # reverts to the frozen formulas' own stochastic jitter, which 2610a69
            # proved is load-bearing (killing it REGRESSED 0.7891->0.7852).
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
                # clash repair stays on the INDEPENDENT shared rng (safety valve)
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





# --- composition-group helpers (reduced-formula bucketing) ---
def _reduced_formula_key(atomic_numbers: np.ndarray) -> tuple:
    """Reduced-formula composition key matching metre `_composition_key`."""
    counts = np.bincount(atomic_numbers.astype(np.int64), minlength=119).astype(np.int64)
    nz = counts[counts > 0]
    if nz.size == 0:
        return ()
    g = int(_reduce(_gcd, (int(x) for x in nz)))
    if g > 1:
        counts = counts // g
    return tuple(int(x) for x in counts)


def compute_group_ids(records) -> list[int]:
    """Map each record index -> integer composition-group id (reduced formula)."""
    key_to_gid: dict[tuple, int] = {}
    gids: list[int] = []
    for r in records:
        k = _reduced_formula_key(r["atomic_numbers"].numpy())
        if k not in key_to_gid:
            key_to_gid[k] = len(key_to_gid)
        gids.append(key_to_gid[k])
    return gids

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="mp_20",
                        choices=sorted(T.DATASETS),
                        help="Dataset name; selects MAX_N and precompute/data stems.")
    parser.add_argument("--greedy", action="store_true",
                        help="MAP/argmax token decoding (zeroed Gumbel noise). "
                             "Scope: all rows when not --sg_stratify, else only "
                             "each SG group's anchor.")
    parser.add_argument("--sg_stratify", action="store_true",
                        help="SG-stratified sampling: per composition, assign "
                             "distinct top space groups across its generations "
                             "(polymorph coverage).")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the trained EMA checkpoint (best.pt).")
    parser.add_argument("--ref", type=str, default=None,
                        help="Reference .pt under --data_dir to sample one CIF "
                             "per record. Default <dataset>_test.pt.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir for CIFs. Default runs/<ckpt stem>/samples.")
    parser.add_argument("--steps", type=int, default=SAMPLE_STEPS)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, sample only the first N reference records.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert args.dataset == T.DATASET, (
        f"dataset mismatch: train import-time {T.DATASET!r} vs {args.dataset!r}")

    script_dir = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_file = args.ref or f"{args.dataset}_test.pt"
    ref_path = Path(args.data_dir)
    if not ref_path.is_absolute():
        ref_path = script_dir / args.data_dir
    ref_path = ref_path / ref_file
    if not ref_path.is_file():
        sys.exit(f"[fatal] missing reference file {ref_path}")

    decode_tag = ("greedy" if args.greedy else "stochastic") + \
                 ("_sgstratify" if args.sg_stratify else "")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = script_dir / "runs" / Path(args.ckpt).stem / f"{decode_tag}_samples"

    # Wyckoff vocab sizes the embedding/head shapes to match the checkpoint.
    wyk_index, wyk_cache, W_RAW = T.load_wyckoff_artifacts(script_dir, args.dataset)
    WYK_VOCAB = WYK_SPECIAL + W_RAW
    SG_VOCAB = N_SG + 1
    print(f"[init] dataset={args.dataset} greedy={args.greedy} "
          f"sg_stratify={args.sg_stratify} device={device}", flush=True)
    print(f"[init] wyckoff: W_RAW={W_RAW} WYK_VOCAB={WYK_VOCAB} SG_VOCAB={SG_VOCAB}", flush=True)

    model = MaskedDiffusionModel(WYK_VOCAB, SG_VOCAB).to(device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"[fatal] missing checkpoint {ckpt_path}")
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=False))
    model.eval()
    print(f"[model] loaded EMA checkpoint from {ckpt_path}", flush=True)

    ref_records = torch.load(str(ref_path), weights_only=False)
    n_full = len(ref_records)
    if args.limit and args.limit > 0:
        ref_records = ref_records[:args.limit]
    print(f"[data] ref={ref_file} n_ref={n_full} sampling={len(ref_records)} "
          f"(limit={args.limit})", flush=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    with torch.no_grad():
        generate_cifs(model, ref_records, out_dir, device, args.steps, rng,
                      greedy=args.greedy, sg_stratify=args.sg_stratify)
    n_cif = len(list(Path(out_dir).glob("*.cif")))
    print(f"[done] wrote {n_cif} CIFs into {out_dir} in {time.time()-t0:.1f}s", flush=True)
    print(f"[next] score with:  python evaluate.py --samples_dir {out_dir} "
          f"--dataset {args.dataset}", flush=True)


if __name__ == "__main__":
    main()
