"""Score a MaskGXT submission from a directory of CIF samples.

``sample.py`` writes one CIF per reference entry to ``<out_dir>/{idx:05d}.cif``,
indexed by the reference ``.pt`` split order. This script reports two metric
families at fixed tolerances (``stol=0.5, ltol=0.3, angle_tol=10.0``):

  * METRe / cRMSE (composition-pooled; paper Table 2).
  * one-to-one match rate + RMSE (index-aligned; paper Table 1), both
    ``Unfiltered`` and ``Filtered`` (drops generations failing CDVAE validity).

Filtered match rate requires ``smact==2.6``.

Usage:
    # by dataset name (reference = test split by default)
    python evaluate.py --samples_dir runs/<run>/samples --dataset mp_20

    # or by explicit reference file
    python evaluate.py --samples_dir runs/<run>/samples --test_file mp_20_test.pt
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, reduce
from math import gcd
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm
from collections import Counter

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser

# smact is only needed for the validity-filtered match rate; defer the import
# error to point-of-use so the other metrics work without it. Pin smact==2.6.
try:
    import itertools

    import smact
    from smact.screening import pauling_test

    _SMACT_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover - environment dependent
    smact = None
    pauling_test = None
    _SMACT_IMPORT_ERROR = _e


# --- METRe match rate (and diagnostic RMSD/cRMSE) ----------------------------
# Slim port of OMatG `omg/analysis/analysis.py::metre_rmsds` and its helpers,
# commit fcb9ba2c2cfd70505b0f142a5b3c44944d78e7f0 (MIT License). Strip-down:
#
#   - We rank by the METRe match rate (single metric, higher is better, in
#     [0, 1]) -- the fraction of reference structures with at least one matching
#     generation under pymatgen StructureMatcher. cRMSE and mean RMSD are
#     diagnostics.
#   - We DROP `ValidAtoms`'s SMACT / CrystalNN / Magpie fingerprint validation
#     (the heavy `smact` / `matminer` deps): the METRe match rate does not
#     condition on validity, so we never need those checks.
#   - StructureMatcher tolerances default to the track values
#     (stol=0.5, ltol=0.3, angle_tol=10.0), matching the METRe paper
#     (arXiv 2509.12178 section 5). Do NOT change them for reported numbers.
#   - Performance: refs are pre-grouped by a hashable composition key, so each
#     gen is matched only against same-composition refs (O(1) bucket lookup)
#     instead of an O(n_ref) elementwise composition scan. Numerically identical
#     to the canonical OMatG implementation; the bucket index is the only change.
#
# Protocol: for each test composition C with multiplicity K_C in the reference
# set, the model generates K_C structures conditioned on C. Total generated
# count == reference count. For every reference structure we find the smallest
# pymatgen RMS distance over generated structures (within the same composition);
# references with no match contribute `stol` to the mean.

_MAX_Z = 119  # H..Og; bincount upper bound for hashable composition key


class CrystalRecord:
    """Pair of (pymatgen Structure, atomic-number array) used by METRe.

    The `valid` flag is kept in the API for parity with OMatG's `ValidAtoms`
    but is always True here; we don't compute SMACT/CrystalNN validity.
    """

    __slots__ = ("structure", "numbers", "valid")

    def __init__(self, structure: Structure, atomic_numbers: Sequence[int]):
        self.structure = structure
        self.numbers = np.asarray(atomic_numbers, dtype=np.int64)
        self.valid = True


def _composition_key(numbers: np.ndarray, check_reduced: bool = True) -> tuple:
    """Hashable composition key: two structures collide iff they share a
    composition. Used to bucket refs so each gen is matched only against
    same-composition candidates (O(1) lookup) instead of an O(n_ref) scan.

    check_reduced=True:  bincount divided by gcd-of-nonzero-counts (canonical
                         reduced formula form), so e.g. NaCl and Na2Cl2 collide.
    check_reduced=False: sorted tuple of atomic numbers (exact composition).
    """
    if check_reduced:
        counts = np.bincount(numbers, minlength=_MAX_Z).astype(np.int64)
        nz = counts[counts > 0]
        if nz.size == 0:
            return ()  # empty record -- degenerate, but hashable
        g = int(reduce(gcd, (int(x) for x in nz)))
        if g > 1:
            counts = counts // g
        return tuple(int(x) for x in counts)
    return tuple(int(x) for x in np.sort(numbers))


def _rms_dist(s_a: Structure, s_b: Structure, ltol: float, stol: float, angle_tol: float) -> Optional[float]:
    """Return normalized RMS distance from pymatgen StructureMatcher, or None on no match."""
    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    res = sm.get_rms_dist(s_a, s_b)
    return None if res is None else float(res[0])


def _match_one_to_many_indexed(
    gen_record: CrystalRecord,
    ref_records: Sequence[CrystalRecord],
    ref_by_key: dict,
    ltol: float,
    stol: float,
    angle_tol: float,
    check_reduced: bool,
) -> list[tuple[float, int]]:
    """For one generated record, return list of (rmsd, ref_index) for matches.

    Uses the precomputed `ref_by_key` index to consider only refs with a
    matching composition, instead of comparing against every ref.
    """
    out: list[tuple[float, int]] = []
    key = _composition_key(gen_record.numbers, check_reduced)
    for idx in ref_by_key.get(key, ()):  # () -> empty tuple, no candidates
        ref = ref_records[idx]
        rms = _rms_dist(gen_record.structure, ref.structure, ltol, stol, angle_tol)
        if rms is not None:
            out.append((rms, idx))
    return out


def _best_rmsd_per_ref(
    gen_list: Sequence[CrystalRecord],
    ref_list: Sequence[CrystalRecord],
    *,
    ltol: float,
    stol: float,
    angle_tol: float,
    num_workers: Optional[int],
    check_reduced: bool,
    enable_progress_bar: bool,
    desc: str,
) -> list[Optional[float]]:
    """Return, for each reference index, the smallest matching RMSD across
    `gen_list` (or None if no generated structure matched).

    Shared inner loop for `metre_metrics`.
    """
    if len(gen_list) > len(ref_list):
        raise ValueError(
            f"len(gen_list)={len(gen_list)} cannot exceed len(ref_list)={len(ref_list)}."
        )

    # Bucket refs by composition key -- O(n_ref) preprocessing, then
    # constant-time candidate lookup per gen.
    ref_records = list(ref_list)
    ref_by_key: dict = {}
    for i, ref in enumerate(ref_records):
        ref_by_key.setdefault(_composition_key(ref.numbers, check_reduced), []).append(i)

    func = partial(
        _match_one_to_many_indexed,
        ref_records=ref_records,
        ref_by_key=ref_by_key,
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        check_reduced=check_reduced,
    )

    best_rmsd: list[Optional[float]] = [None] * len(ref_list)

    def _absorb(matches: list[tuple[float, int]]) -> None:
        for rmsd, ref_idx in matches:
            cur = best_rmsd[ref_idx]
            if cur is None or rmsd < cur:
                best_rmsd[ref_idx] = rmsd

    if num_workers is None or num_workers <= 1:
        iterator = tqdm(
            gen_list, total=len(gen_list), desc=desc,
            disable=not enable_progress_bar,
        )
        for gen in iterator:
            _absorb(func(gen))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(func, gen): i for i, gen in enumerate(gen_list)}
            iterator = tqdm(
                as_completed(futures), total=len(futures), desc=desc,
                disable=not enable_progress_bar,
            )
            for fut in iterator:
                _absorb(fut.result())

    return best_rmsd


def metre_metrics(
    gen_list: Sequence[CrystalRecord],
    ref_list: Sequence[CrystalRecord],
    *,
    ltol: float = 0.3,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    num_workers: Optional[int] = None,
    check_reduced: bool = True,
    enable_progress_bar: bool = True,
    desc: str = "metre",
) -> dict:
    """Run a single StructureMatcher pass and return all aggregate scalars
    derivable from it. `match_rate` (the METRe match rate) is the primary
    metric; the others are diagnostic -- they help disentangle "matched but
    imprecise" from "did not match".

    Returns dict with keys (each a Python float / int):
        match_rate   n_matched / n_total -- METRe match rate, range [0, 1]
        mean_rmsd    mean over matched refs of best RMSD -- NaN if 0 matches
        cRMSE        mean over refs of (best RMSD or stol if no match)
        n_matched    int, refs with at least one matching gen
        n_total      int, len(ref_list)
    """
    best_rmsd = _best_rmsd_per_ref(
        gen_list, ref_list,
        ltol=ltol, stol=stol, angle_tol=angle_tol,
        num_workers=num_workers, check_reduced=check_reduced,
        enable_progress_bar=enable_progress_bar, desc=desc,
    )
    n_total = len(best_rmsd)
    matched = [r for r in best_rmsd if r is not None]
    n_matched = len(matched)
    crmse = float(np.mean([r if r is not None else stol for r in best_rmsd]))
    mean_rmsd = float(np.mean(matched)) if n_matched > 0 else float("nan")
    return {
        "cRMSE": crmse,
        "match_rate": n_matched / n_total,
        "mean_rmsd": mean_rmsd,
        "n_matched": n_matched,
        "n_total": n_total,
    }


def _ref_from_record(rec) -> CrystalRecord:
    """Build a reference CrystalRecord from a ground-truth split record."""
    frac = rec["frac_coords"].numpy()
    lat = rec["lattice"].numpy()
    species = rec["atomic_numbers"].numpy()
    return CrystalRecord(
        Structure(Lattice(lat), species, frac, coords_are_cartesian=False),
        species,
    )


def _struct_from_cif(path: Path) -> Structure:
    """Parse a generated CIF into a pymatgen Structure (raises on bad CIFs)."""
    return CifParser(str(path)).parse_structures(primitive=False)[0]


def _gen_from_cif(path: Path) -> CrystalRecord:
    """Build a generated METRe CrystalRecord from a CIF file."""
    struct = _struct_from_cif(path)
    return CrystalRecord(struct, struct.atomic_numbers)


# --- one-to-one CSP match rate (paper Table 1) -------------------------------
# Index-aligned matching (gen i vs ref i), unfiltered and validity-filtered.
# Vendored from crystalite (FlowMM / CDVAE lineage): ``smact_validity`` /
# ``structure_validity`` / ``Crystal`` (src/eval/crystal.py) and ``RecEval``
# (src/eval/csp_eval.py).

_MATCH_MAX_Z = 94  # CDVAE atom-type upper bound for validity

# Element symbol per atomic number (index 0 = "X" placeholder).
_CHEMICAL_SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def _smact_validity(comp, count, use_pauling_test=True, include_alloys=True,
                    allow_missing_ox_states=False):
    """SMACT charge-balance + electronegativity validity of a composition."""
    if smact is None:
        raise ImportError(
            f"smact required for the filtered match rate ({_SMACT_IMPORT_ERROR}); "
            "install `smact==2.6`."
        )
    elem_symbols = tuple([_CHEMICAL_SYMBOLS[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = []
    for elem in smact_elems:
        ox_states = elem.oxidation_states
        if not ox_states:
            return True if allow_missing_ox_states else False
        ox_combos.append(ox_states)
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        cn_e, cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def _structure_validity(crystal, cutoff=0.5):
    """No two atoms closer than ``cutoff`` A and volume >= 0.1 (CDVAE)."""
    dist_mat = crystal.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    return True


class MatchCrystal:
    """A pymatgen Structure plus a CDVAE ``valid`` flag (comp and struct).

    ``compute_validity=False`` keeps ``valid=True`` (unfiltered score, and
    references, which are ground truth).
    """

    __slots__ = ("structure", "atom_types", "valid", "comp_valid", "struct_valid")

    def __init__(self, structure: Structure, atomic_numbers: Sequence[int],
                 compute_validity: bool = False):
        self.structure = structure
        self.atom_types = np.asarray(atomic_numbers, dtype=np.int64)
        if not compute_validity:
            self.valid = self.comp_valid = self.struct_valid = True
            return
        if (self.atom_types < 1).any() or (self.atom_types > _MATCH_MAX_Z).any():
            self.valid = self.comp_valid = self.struct_valid = False
            return
        self.struct_valid = _structure_validity(structure)
        elem_counter = Counter(int(a) for a in self.atom_types)
        elems = sorted(elem_counter.keys())
        counts = np.array([elem_counter[e] for e in elems])
        counts = counts // np.gcd.reduce(counts)
        try:
            self.comp_valid = _smact_validity(tuple(elems), tuple(counts.tolist()))
        except ImportError:
            raise
        except Exception:
            self.comp_valid = False
        self.valid = self.comp_valid and self.struct_valid


class RecEval:
    """One-to-one (index-aligned) match rate + RMSE (crystalite ``RecEval``).

    ``preds[i]`` is the single candidate for reference ``gts[i]``; matched iff
    both are ``valid`` and StructureMatcher returns an RMS distance.
    ``match_rate`` is over all references (missing/invalid count as misses).
    """

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10.0, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def process_one(self, pred, gt, is_valid):
        if not is_valid:
            return None
        try:
            rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
            if rms_dist is None:
                return None
            return rms_dist[0]
        except Exception:
            return None

    def get_match_rate_and_rms(self, enable_progress_bar=True, desc="match"):
        rms_dists = []
        it = tqdm(range(len(self.preds)), desc=desc, disable=not enable_progress_bar)
        for i in it:
            p = self.preds[i]
            if p is None:
                rms_dists.append(None)
                continue
            is_valid = p.valid and self.gts[i].valid
            rms_dists.append(self.process_one(p, self.gts[i], is_valid))
        rms_dists = np.array(rms_dists, dtype=object)
        n_total = len(self.preds)
        matched = rms_dists[rms_dists != None]  # noqa: E711 (None sentinel)
        n_matched = len(matched)
        mean_rms = float(matched.astype(float).mean()) if n_matched > 0 else float("nan")
        return {
            "match_rate": n_matched / n_total,
            "rms_dist": mean_rms,
            "n_matched": int(n_matched),
            "n_total": int(n_total),
        }


def _one_to_one_metrics(
    gen_structs: Sequence[Optional[Structure]],
    ref_records: Sequence,
    *,
    ltol: float = 0.3,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    enable_progress_bar: bool = True,
) -> dict:
    """One-to-one match rate + RMSE, unfiltered and filtered.

    ``gen_structs[i]`` (Structure or None) is the single candidate for
    ``ref_records[i]``. Returns ``unfiltered`` and ``filtered`` sub-dicts.
    """
    # References are ground truth: always valid.
    gt_crys = [
        MatchCrystal(
            Structure(Lattice(r["lattice"].numpy()), r["atomic_numbers"].numpy(),
                      r["frac_coords"].numpy(), coords_are_cartesian=False),
            r["atomic_numbers"].numpy(),
            compute_validity=False,
        )
        for r in ref_records
    ]

    # Generations: build once per mode (None -> None, an automatic miss).
    def _build(compute_validity: bool):
        out = []
        for s in gen_structs:
            if s is None:
                out.append(None)
            else:
                out.append(MatchCrystal(s, s.atomic_numbers,
                                        compute_validity=compute_validity))
        return out

    print("[1to1] computing one-to-one match rate (unfiltered + filtered) ...")
    unfiltered = RecEval(
        _build(compute_validity=False), gt_crys,
        stol=stol, angle_tol=angle_tol, ltol=ltol,
    ).get_match_rate_and_rms(enable_progress_bar=enable_progress_bar, desc="unfiltered")
    filtered = RecEval(
        _build(compute_validity=True), gt_crys,
        stol=stol, angle_tol=angle_tol, ltol=ltol,
    ).get_match_rate_and_rms(enable_progress_bar=enable_progress_bar, desc="filtered")
    return {"unfiltered": unfiltered, "filtered": filtered}


def compute_metrics(samples_dir, ref_file, *, data_dir="./data", limit=0,
                    num_workers=0, enable_progress_bar=True) -> dict:
    """Load CIF samples + reference split and compute every metric in one pass.

    Single source of truth for the numbers: ``main`` prints what this returns,
    and ``train.py`` calls it directly (no subprocess, no stdout parsing). A
    failed measurement raises here rather than degrading to a silent sentinel.

    Returns the METRe keys (``match_rate``, ``cRMSE``, ``mean_rmsd``,
    ``n_matched``, ``n_total``) plus ``one_to_one`` (unfiltered/filtered) and
    ``ref_file``. Raises FileNotFoundError (missing samples_dir / reference) or
    ValueError (no CIF could be loaded).
    """
    samples_dir = Path(samples_dir)
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"samples_dir not found: {samples_dir}")

    # Load reference split (ground truth used only as reference for matching).
    ref_path = Path(data_dir) / ref_file
    if not ref_path.is_file():
        raise FileNotFoundError(f"reference file not found: {ref_path}")
    ref_records = torch.load(str(ref_path), weights_only=False)
    if limit > 0:
        ref_records = ref_records[:limit]
    n = len(ref_records)
    print(f"[data] reference split: {n} structures (file={ref_file})")

    print("[ref] building reference pymatgen structures ...")
    ref_list = [_ref_from_record(r)
                for r in tqdm(ref_records, desc="ref", disable=not enable_progress_bar)]

    # Parse each CIF once, kept at its index (None if missing/unparseable);
    # consumed by both METRe and the index-aligned match rate. We don't abort
    # on per-CIF failures -- collect and report.
    print(f"[gen] loading CIFs from {samples_dir} ...")
    gen_structs: list[Optional[Structure]] = [None] * n  # index-aligned
    missing: list[int] = []           # file doesn't exist
    unparseable: list[tuple[int, str]] = []  # (idx, short reason)
    for i in range(n):
        path = samples_dir / f"{i:05d}.cif"
        if not path.exists():
            missing.append(i)
            continue
        try:
            gen_structs[i] = _struct_from_cif(path)
        except Exception as e:
            unparseable.append((i, type(e).__name__ + ": " + str(e)[:120]))
    n_dropped = len(missing) + len(unparseable)
    if missing:
        print(f"[gen] missing {len(missing)} CIFs (first few: {missing[:5]})")
    if unparseable:
        print(f"[gen] unparseable {len(unparseable)} CIFs "
              f"(first few: {[idx for idx, _ in unparseable[:5]]})")
        for idx, reason in unparseable[:5]:
            print(f"[gen]   {idx:05d}.cif: {reason}")
    n_loaded = n - n_dropped
    print(f"[gen] loaded gens: {n_loaded}/{n} "
          f"({n_dropped} dropped, {100*n_dropped/n:.2f}%)")

    if n_loaded == 0:
        raise ValueError(
            f"all {n} CIFs in {samples_dir} missing or unparseable -- nothing to score."
        )

    # METRe pool: composition-bucketed CrystalRecords for the loaded gens.
    gen_list = [
        CrystalRecord(s, s.atomic_numbers) for s in gen_structs if s is not None
    ]

    # Compute METRe at the fixed tolerances.
    print(f"[metre] computing METRe (stol=0.5, ltol=0.3, angle_tol=10.0, "
          f"workers={num_workers or 'seq'}) ...")
    m = metre_metrics(
        gen_list,
        ref_list,
        ltol=0.3,
        stol=0.5,
        angle_tol=10.0,
        num_workers=num_workers if num_workers > 0 else None,
        check_reduced=True,
        enable_progress_bar=enable_progress_bar,
        desc="metre",
    )

    # One-to-one match rate (unfiltered + validity-filtered, paper Table 1).
    one2one = _one_to_one_metrics(gen_structs, ref_records,
                                  enable_progress_bar=enable_progress_bar)

    return {**m, "one_to_one": one2one, "ref_file": ref_file}


def _print_report(result: dict) -> None:
    """Print the human-readable metric block for ``compute_metrics`` output."""
    m = result
    one2one = result["one_to_one"]
    print("=" * 60)
    print(f"MaskGXT evaluation  (n={m['n_total']}, file={result['ref_file']})")
    print("-" * 60)
    print("METRe (composition-pooled coverage):")
    print(f"  match rate (higher is better) = {m['match_rate']*100:.2f}%   "
          f"({m['n_matched']}/{m['n_total']})")
    print(f"  mean RMSD over matched only   = {m['mean_rmsd']:.4f}")
    print(f"  cRMSE (lower is better)       = {m['cRMSE']:.4f}")
    print("-" * 60)
    print("One-to-one CSP match rate (index-aligned, paper Table 1):")
    uf, fl = one2one["unfiltered"], one2one["filtered"]
    print(f"  Unfiltered  match rate = {uf['match_rate']*100:6.2f}%   "
          f"RMSE = {uf['rms_dist']:.4f}   ({uf['n_matched']}/{uf['n_total']})")
    print(f"  Filtered    match rate = {fl['match_rate']*100:6.2f}%   "
          f"RMSE = {fl['rms_dist']:.4f}   ({fl['n_matched']}/{fl['n_total']})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples_dir", type=str, required=True,
                        help="Directory containing {idx:05d}.cif files.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (mp_20, mp_20_ps, mpts_52). With "
                             "--split, selects the reference file "
                             "<dataset>_<split>.pt. Ignored if --test_file given.")
    parser.add_argument("--split", type=str, default="test",
                        choices=("train", "val", "test"),
                        help="Reference split when using --dataset (default test).")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Reference .pt file under --data_dir. Overrides "
                             "--dataset/--split. (train.py passes this during "
                             "validation.)")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, evaluate on first N reference entries only (debug).")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Process pool size for StructureMatcher (0 = sequential, "
                             "default). Sequential is fastest in practice: the parallel "
                             "path pickles the full reference list per task and IPC "
                             "overhead exceeds StructureMatcher CPU cost.")
    args = parser.parse_args()

    # Resolve the reference file.
    if args.test_file is not None:
        ref_file = args.test_file
    elif args.dataset is not None:
        ref_file = f"{args.dataset}_{args.split}.pt"
    else:
        raise SystemExit("provide --dataset (and optional --split) or --test_file")

    try:
        result = compute_metrics(
            args.samples_dir, ref_file,
            data_dir=args.data_dir, limit=args.limit, num_workers=args.num_workers,
        )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))
    _print_report(result)
    return result


if __name__ == "__main__":
    main()
