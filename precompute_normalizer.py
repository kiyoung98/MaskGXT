"""precompute_normalizer.py — per-space-group Euclidean-normalizer table for
ONLINE, recompute-free Wyckoff-token augmentation.

For each of the 230 space groups, pyxtal `Group(sg).get_alternatives()` returns
the Euclidean-normalizer cosets: each coset is an affine operation (R, t) plus
the induced Wyckoff-LETTER permutation. A TRANSLATION coset (R == I) is exactly
an origin shift by t that relabels Wyckoff letters by the permutation (e.g.
Fm-3m: t=(1/2,1/2,1/2) swaps 4a<->4b). Applying (shift coords by t, relabel
tokens by the permutation) keeps the (coord, Wyckoff-token) pair self-consistent
WITHOUT any per-shift spglib recompute -- so the augmentation runs fully online
at train time from this tiny static table (a few KB), replacing the 45 MB
per-crystal shift_aug.pt.

Writes:
    precompute/normalizer.pt  -- dict[int sg] -> {
        "translations": list[np.ndarray(3)]  coset shift vectors (incl identity=0),
        "perms":        list[dict[str,str]]  old_letter -> new_letter per coset,
        "n_translation_cosets": int,
        "n_nontranslation": int            # cosets with R != I (skipped; logged)
    }
Only TRANSLATION cosets (R == I) are stored -- those are pure origin shifts that
leave the lattice untouched. Non-translation cosets (rotations/scalings) are
counted but skipped (they would rotate the cell, interfering with the lattice
tokens); identity is always index 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("precompute")
N_SG = 230


def _parse_op(xyz: str):
    """Parse a 'x+1/2,y,z' style coset representative into (R 3x3, t 3)."""
    from pymatgen.core.operations import SymmOp
    op = SymmOp.from_xyz_str(xyz)
    R = np.array(op.rotation_matrix, dtype=float)
    t = np.array(op.translation_vector, dtype=float) % 1.0
    return R, t


def main():
    from pyxtal.symmetry import Group
    OUT_DIR.mkdir(exist_ok=True)
    table = {}
    tot_trans = tot_nontrans = 0
    swap_sgs = 0
    for sg in range(1, N_SG + 1):
        alt = Group(sg).get_alternatives()
        cosets = alt["Coset Representative"]
        wps = alt["Transformed WP"]
        ref = wps[0].split()                      # identity letter order 'a b c ...'
        translations, perms = [], []
        n_nontrans = 0
        seen = set()
        for cos, wp in zip(cosets, wps):
            R, t = _parse_op(cos)
            if not np.allclose(R, np.eye(3)):
                n_nontrans += 1
                continue
            new = wp.split()
            perm = {old: nw for old, nw in zip(ref, new)}
            key = (tuple(np.round(t, 6)), tuple(sorted(perm.items())))
            if key in seen:
                continue
            seen.add(key)
            translations.append(t.astype(np.float32))
            perms.append(perm)
        # ensure identity (t=0, perm=identity) is index 0
        id_idx = next((i for i, t in enumerate(translations) if np.allclose(t, 0)), None)
        if id_idx is None:
            translations.insert(0, np.zeros(3, np.float32))
            perms.insert(0, {l: l for l in ref})
        elif id_idx != 0:
            translations.insert(0, translations.pop(id_idx))
            perms.insert(0, perms.pop(id_idx))
        table[sg] = {
            "translations": translations,
            "perms": perms,
            "n_translation_cosets": len(translations),
            "n_nontranslation": n_nontrans,
        }
        tot_trans += len(translations)
        tot_nontrans += n_nontrans
        if any(any(k != v for k, v in p.items()) for p in perms):
            swap_sgs += 1

    import torch
    torch.save(table, OUT_DIR / "normalizer.pt")
    summary = [
        "precompute_normalizer.py self-verify report",
        f"space groups                       : {N_SG}",
        f"total translation cosets (incl id) : {tot_trans}  (mean {tot_trans/N_SG:.2f}/SG)",
        f"total non-translation cosets (skip): {tot_nontrans}",
        f"SGs whose translation cosets RELABEL Wyckoff letters: {swap_sgs}",
        f"example SG225: n_cosets={table[225]['n_translation_cosets']}, "
        f"perm[1]={ {k:v for k,v in table[225]['perms'][1].items() if k!=v} }",
    ]
    (OUT_DIR / "normalizer_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)
    assert tot_trans >= N_SG, "every SG must have >=1 (identity) coset"


if __name__ == "__main__":
    main()
