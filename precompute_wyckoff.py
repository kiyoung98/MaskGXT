"""
precompute_wyckoff.py — Wyckoff/SG annotation for the Wyckoff-Hybrid tokenizer.

Run as:
    python precompute_wyckoff.py --dataset mp_20

Reads:
    data/<dataset>_train.pt   (list[dict]: frac_coords, lattice, atomic_numbers)

Writes:
    precompute/wyckoff_index.pt              global (sg, letter) -> int (pyxtal);
                                             dataset-independent, written once
    precompute/wyckoff_cache_<dataset>.pt    per-crystal {sg, wyckoff_tokens,
                                             equivalent_atoms} for this dataset
    precompute/wyckoff_summary_<dataset>.txt human-readable self-verify report

Token-id convention (the cache stores RAW joint (sg,letter) indices in
[0, W) for known pairs, or -1 for UNK). train.py maps these to its own
embedding ids by:
    WYK_MASK = 0
    WYK_UNK  = 1
    raw_tok (>=0)  ->  raw_tok + 2
    raw_tok == -1  ->  WYK_UNK
SG ids in the cache are raw 1..230; train.py maps:
    SG_MASK = 0, SG_UNK = 0 not used; sg in 1..230 -> sg (id==sg), SG mask id = 0.
(So SG embedding is Embedding(231): id 0 = MASK, 1..230 = space groups.)

This is the SINGLE coherent change's precompute. Self-verifies per
contract.md and asserts fail-fast on degeneracy.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import torch
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from configs import DATASETS

warnings.filterwarnings("ignore")

OUT_DIR = Path("precompute")
SYMPREC_SEQ = (0.01, 0.05, 0.1)
RARE_THRESHOLD = 3


def build_global_wyckoff_index() -> dict:
    try:
        from pyxtal import Group
    except ImportError:
        sys.exit("pyxtal not installed. Run: $PY -m pip install pyxtal")
    index: dict = {}
    for sg in range(1, 231):
        try:
            g = Group(sg)
            for wp in g.Wyckoff_positions:
                key = (sg, wp.letter)
                if key not in index:
                    index[key] = len(index)
        except Exception as e:
            print(f"  warning: pyxtal failed for SG {sg}: {e}")
            continue
    return index


def analyse_one(crystal, index):
    frac = crystal["frac_coords"].numpy()
    lat = crystal["lattice"].numpy()
    Z = crystal["atomic_numbers"].tolist()
    n = len(Z)
    struct = Structure(Lattice(lat), Z, frac, coords_are_cartesian=False)
    for symprec in SYMPREC_SEQ:
        try:
            sga = SpacegroupAnalyzer(struct, symprec=symprec)
            sg = sga.get_space_group_number()
            sym = sga.get_symmetry_dataset()
            letters = list(sym["wyckoffs"])
            equiv = [int(x) for x in sym["equivalent_atoms"]]
            toks = [index.get((sg, l), -1) for l in letters]
            return int(sg), toks, equiv
        except Exception:
            continue
    return 1, [-1] * n, list(range(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=str, default="mp_20", choices=sorted(DATASETS),
                    help="Dataset name; reads data/<dataset>_train.pt.")
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    data_path = args.data_dir / f"{args.dataset}_train.pt"
    cache_path = OUT_DIR / f"wyckoff_cache_{args.dataset}.pt"
    summary_path = OUT_DIR / f"wyckoff_summary_{args.dataset}.txt"
    if not data_path.exists():
        sys.exit(f"missing {data_path}")
    OUT_DIR.mkdir(exist_ok=True)

    crystals = torch.load(data_path, weights_only=False)
    n_crystals = len(crystals)
    print(f"loaded {n_crystals} train crystals")

    index = build_global_wyckoff_index()
    print(f"global Wyckoff index size = {len(index)}")
    torch.save(index, OUT_DIR / "wyckoff_index.pt")

    cache = []
    label_counter: Counter = Counter()
    sg_counter: Counter = Counter()
    fail_count = 0
    unk_atoms = 0
    total_atoms = 0
    token_copy_violations = 0

    for i, crystal in enumerate(crystals):
        sg, toks, equiv = analyse_one(crystal, index)
        sg_counter[sg] += 1
        n = crystal["atomic_numbers"].numel()
        total_atoms += n
        if sg == 1 and n > 1 and all(t < 0 for t in toks):
            fail_count += 1
        for t in toks:
            if t >= 0:
                label_counter[t] += 1
            else:
                unk_atoms += 1

        # token-copy verification: same orbit -> same token
        orbit_to_tok = {}
        for orbit_id, tok in zip(equiv, toks):
            if orbit_id not in orbit_to_tok:
                orbit_to_tok[orbit_id] = tok
            elif orbit_to_tok[orbit_id] != tok:
                token_copy_violations += 1

        cache.append({"sg": sg, "wyckoff_tokens": toks, "equivalent_atoms": equiv})
        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1} / {n_crystals}")

    rare_tokens = {t for t, c in label_counter.items() if c < RARE_THRESHOLD}
    crystals_with_rare = sum(
        1 for c in cache if any(t in rare_tokens for t in c["wyckoff_tokens"])
    )

    torch.save(cache, cache_path)

    fail_rate = fail_count / max(n_crystals, 1)
    unk_rate = unk_atoms / max(total_atoms, 1)
    lines = [
        "precompute_wyckoff.py self-verify report",
        "========================================",
        f"crystals processed         : {n_crystals}",
        f"global Wyckoff vocab size  : {len(index)}",
        f"P1-fallback rate           : {fail_rate:.2%}  (target < 5%)",
        f"UNK atom rate              : {unk_rate:.2%}",
        f"token-copy violations      : {token_copy_violations}  (must be 0)",
        f"distinct Wyckoff tokens hit: {len(label_counter)} / {len(index)}",
        f"rare (<{RARE_THRESHOLD}) tokens     : {len(rare_tokens)} "
        f"({len(rare_tokens)/max(len(label_counter),1):.1%} of hit)",
        f"crystals with any rare W   : {crystals_with_rare}",
        f"top-5 SGs                  : {sg_counter.most_common(5)}",
        f"top-10 Wyckoff tokens      : {label_counter.most_common(10)}",
    ]
    summary = "\n".join(lines)
    summary_path.write_text(summary + "\n")
    print()
    print(summary)

    assert fail_rate < 0.20, f"too many P1 fallbacks: {fail_rate:.2%}"
    assert token_copy_violations == 0, "token-copying violated"
    assert 1500 <= len(index) <= 2000, f"vocab size {len(index)} out of range"
    print("\nself-verify PASSED")


if __name__ == "__main__":
    main()
