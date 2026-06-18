"""Score a MaskGXT submission from a directory of CIF samples.

``sample.py`` (and ``train.py`` during validation) writes one CIF per reference
entry to ``<out_dir>/{idx:05d}.cif``, indexed by the order of the reference
``.pt`` split. This script loads those CIFs, builds the reference set from the
ground-truth ``.pt`` tensors, and computes the full METRe metric set at the
fixed tolerances (``stol=0.5, ltol=0.3, angle_tol=10.0``).

METRe match rate (higher is better, range [0, 1]) is the primary metric; cRMSE
(lower is better) and mean_rmsd (precision when matched) are reported as
diagnostics.

Usage:
    # by dataset name (reference = test split by default)
    python evaluate.py --samples_dir runs/<run>/samples --dataset mp_20

    # or by explicit reference file
    python evaluate.py --samples_dir runs/<run>/samples --test_file mp_20_test.pt
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, reduce
from math import gcd
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser


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


def _gen_from_cif(path: Path) -> CrystalRecord:
    """Build a generated CrystalRecord from a CIF file.

    Uses pymatgen's default ``occupancy_tolerance``. If a CIF can't be
    parsed (e.g. atoms collapsed to identical fractional coords ->
    occupancy > tolerance, or a lattice axis collapsed below the
    minimum-thickness threshold), this raises; the caller catches and
    skips that gen from the matching pool. METRe's per-composition
    matching makes the absence a graceful "no candidate" rather than a
    hard error.
    """
    struct = CifParser(str(path)).parse_structures(primitive=False)[0]
    return CrystalRecord(struct, struct.atomic_numbers)


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

    samples_dir = Path(args.samples_dir)
    if not samples_dir.is_dir():
        raise SystemExit(f"samples_dir not found: {samples_dir}")

    # Load reference split (ground truth used only as reference for matching).
    ref_path = Path(args.data_dir) / ref_file
    if not ref_path.is_file():
        raise SystemExit(f"reference file not found: {ref_path}")
    ref_records = torch.load(str(ref_path), weights_only=False)
    if args.limit > 0:
        ref_records = ref_records[:args.limit]
    n = len(ref_records)
    print(f"[data] reference split: {n} structures (file={ref_file})")

    print("[ref] building reference pymatgen structures ...")
    ref_list = [_ref_from_record(r) for r in tqdm(ref_records, desc="ref")]

    # Load CIFs at fixed index order. METRe matching is composition-based,
    # not index-based -- a missing or unparseable gen at index i simply
    # reduces the pool of candidates for refs of that composition. Other
    # refs of the same composition can still match against the remaining
    # gens. So we don't abort here; we collect skips and report.
    print(f"[gen] loading CIFs from {samples_dir} ...")
    gen_list: list[CrystalRecord] = []
    missing: list[int] = []           # file doesn't exist
    unparseable: list[tuple[int, str]] = []  # (idx, short reason)
    for i in range(n):
        path = samples_dir / f"{i:05d}.cif"
        if not path.exists():
            missing.append(i)
            continue
        try:
            gen_list.append(_gen_from_cif(path))
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
    print(f"[gen] loaded gen_list: {len(gen_list)}/{n} "
          f"({n_dropped} dropped, {100*n_dropped/n:.2f}%)")

    if len(gen_list) == 0:
        sys.stderr.write(
            f"[eval] all {n} CIFs missing or unparseable -- nothing to score.\n"
        )
        raise SystemExit(2)

    # Compute METRe at the fixed tolerances.
    print(f"[metre] computing METRe (stol=0.5, ltol=0.3, angle_tol=10.0, "
          f"workers={args.num_workers or 'seq'}) ...")
    m = metre_metrics(
        gen_list,
        ref_list,
        ltol=0.3,
        stol=0.5,
        angle_tol=10.0,
        num_workers=args.num_workers if args.num_workers > 0 else None,
        check_reduced=True,
        enable_progress_bar=True,
        desc="metre",
    )
    print("=" * 60)
    print(f"MaskGXT  METRe match rate (n={n}, higher is better) = "
          f"{m['match_rate']*100:.2f}%   ({m['n_matched']}/{m['n_total']})")
    print(f"         mean RMSD over matched only                = {m['mean_rmsd']:.4f}")
    print(f"         cRMSE (lower is better, in [0, stol=0.5])  = {m['cRMSE']:.4f}")
    print("=" * 60)
    return m


if __name__ == "__main__":
    main()
