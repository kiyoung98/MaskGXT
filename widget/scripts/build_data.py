"""
Build structures.json for the crystal structure explorer widget.
Selects ~100 representative structures from mp_20_ps greedy_sgstratify_samples
grouped by application category, converts to pymatgen JSON format for matterviz.

Usage:
    conda run -n liflow python widget/scripts/build_data.py
    (or from MaskGXT root)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).parent.parent.parent  # MaskGXT/
SAMPLES_DIR = ROOT / "runs/mp_20_ps/greedy_sgstratify_samples"
OUT_FILE = Path(__file__).parent.parent / "static/data/structures.json"


def parse_cif_elems(cif_text: str) -> list[str]:
    lines = cif_text.split("\n")
    in_loop = False
    atoms: list[str] = []
    for line in lines:
        s = line.strip()
        if "_atom_site_type_symbol" in s:
            in_loop = True
            continue
        if in_loop:
            if s.startswith("_") or s.startswith("loop_"):
                if atoms:
                    break
                continue
            parts = s.split()
            if parts and re.match(r"^[A-Z][a-z]?$", parts[0]):
                atoms.append(parts[0])
            elif atoms:
                break
    return atoms


CATEGORIES = [
    {
        "id": "battery",
        "label": "🔋 Battery Cathode",
        "quota": 20,
        "match": lambda es: "Li" in es and "O" in es and bool(es & {"Ni", "Co", "Mn", "Fe", "Ti"}),
    },
    {
        "id": "perovskite",
        "label": "⚡ Perovskite (Magnetic / Ferroelectric / Catalysis)",
        "quota": 15,
        "match": lambda es: "O" in es
            and bool(es & {"Ba", "Sr", "Ca", "La", "Pr", "Nd", "Sm"})
            and bool(es & {"Ti", "Mn", "Fe", "Co", "Cr", "Ni", "V", "Nb", "Mo", "W"})
            and "Cu" not in es,
    },
    {
        "id": "catalyst",
        "label": "⚗️ Noble Metal Catalyst (HER/OER)",
        "quota": 15,
        "match": lambda es: bool(es & {"Ru", "Ir", "Rh", "Pt", "Pd"}) and "O" not in es,
    },
    {
        "id": "magnet",
        "label": "🧲 Rare-Earth Permanent Magnet",
        "quota": 10,
        "match": lambda es: bool(es & {"Nd", "Pr", "Sm", "Dy", "Tb"})
            and bool(es & {"Fe", "Co"})
            and bool(es & {"B", "Si", "Ge", "Sn", "Ga", "Al"}),
    },
    {
        "id": "thermoelectric",
        "label": "🌡️ Thermoelectric",
        "quota": 10,
        "match": lambda es: bool(es & {"Te", "Se"})
            and bool(es & {"Bi", "Pb", "Sb", "Ge", "Sn"})
            and "O" not in es,
    },
    {
        "id": "solid_electrolyte",
        "label": "🔋 Solid Electrolyte",
        "quota": 10,
        "match": lambda es: "Li" in es and "P" in es and "O" in es and bool(es & {"S", "Cl", "F"}),
    },
    {
        "id": "superconductor",
        "label": "🌡️ High-Tc Superconductor (Cuprate)",
        "quota": 10,
        "match": lambda es: "Cu" in es and "O" in es
            and bool(es & {"Ba", "La", "Sr", "Bi", "Tl"})
            and not bool(es & {"Fe", "Mn", "Co", "Cr"}),
    },
    {
        "id": "phosphor",
        "label": "💡 Rare-Earth Phosphor (LED/Display)",
        "quota": 6,
        "match": lambda es: bool(es & {"Eu", "Ce", "Tb"})
            and bool(es & {"Al", "Ga", "Si", "Y", "B"})
            and bool(es & {"O", "N", "F"}),
    },
]

EXTRA_NOTABLE = [
    "02788",  # Bi-S-Se thermoelectric
    "06945",  # Bi-Se
    "03509",  # Co-Ge-Nd magnet
    "03821",  # Al-B-Eu-O phosphor
]


def origin_shift_to_highest_symmetry(struct: Structure) -> Structure:
    """Shift the structure so the highest-symmetry Wyckoff site sits at the origin.

    Strategy:
    1. Get symmetrised structure and Wyckoff symbols via SpacegroupAnalyzer.
    2. Pick the site whose Wyckoff multiplicity is lowest (fewest equivalent
       positions → highest site symmetry). Break ties by choosing the one
       closest to (0,0,0) in fractional coords.
    3. Translate all fractional coords so that site lands exactly at (0,0,0),
       then wrap everything back into [0,1).
    """
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.1)
        sym_struct = sga.get_symmetrized_structure()
        wyckoff_data = sym_struct.equivalent_sites  # list of lists of PeriodicSite

        # Find the Wyckoff set with the smallest multiplicity (highest site symmetry).
        # Tie-break: prefer the set whose representative site has the smallest
        # fractional coordinates (lexicographic on wrapped coords), matching the
        # ITA convention of placing the high-symmetry site at the origin.
        def set_sort_key(sites):
            mult = len(sites)
            rep = sites[0].frac_coords % 1.0
            return (mult, rep[0], rep[1], rep[2])

        best_set = min(wyckoff_data, key=set_sort_key)
        # Use the first site of the best set as the origin anchor.
        anchor = best_set[0]
        shift = anchor.frac_coords % 1.0  # shift in [0,1) so subtraction wraps correctly

        new_sites = []
        for site in struct:
            new_frac = site.frac_coords - shift
            # wrap into [0, 1)
            new_frac = new_frac % 1.0
            new_sites.append((site.species, new_frac))

        shifted = Structure(struct.lattice, [s for s, _ in new_sites], [f for _, f in new_sites])
        return shifted
    except Exception:
        return struct  # fall back to original if symmetry analysis fails


def build_entry(cif_path: Path) -> dict | None:
    cif_text = cif_path.read_text()
    try:
        struct = Structure.from_str(cif_text, fmt="cif")
    except Exception as e:
        print(f"  skip {cif_path.name}: {e}", file=sys.stderr)
        return None

    struct = origin_shift_to_highest_symmetry(struct)

    formula = struct.composition.reduced_formula
    n_atoms = len(struct)
    sg_str = cif_text.split("_symmetry_space_group_name_H-M")[1].strip().split("\n")[0].strip("' ")
    return {
        "id": cif_path.stem,
        "formula": formula,
        "n_atoms": n_atoms,
        "space_group": sg_str,
        "pmg": struct.as_dict(),
        "cif": cif_text,
    }


def main() -> None:
    cif_files = sorted(SAMPLES_DIR.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files in {SAMPLES_DIR}", flush=True)

    # Pre-parse element sets
    print("Parsing element sets...", flush=True)
    parsed: list[tuple[str, set[str]]] = []
    for f in cif_files:
        txt = f.read_text()
        elems = set(parse_cif_elems(txt))
        parsed.append((f.stem, elems))

    used: set[str] = set()
    categories_out = []

    for cat in CATEGORIES:
        entries_out = []
        for stem, elems in parsed:
            if stem in used:
                continue
            if cat["match"](elems):
                cif_path = SAMPLES_DIR / f"{stem}.cif"
                print(f"  [{cat['id']}] {stem} {sorted(elems)}", flush=True)
                entry = build_entry(cif_path)
                if entry:
                    entries_out.append(entry)
                    used.add(stem)
            if len(entries_out) >= cat["quota"]:
                break
        print(f"  → {len(entries_out)} entries for {cat['label']}", flush=True)
        if entries_out:
            categories_out.append({
                "id": cat["id"],
                "label": cat["label"],
                "entries": entries_out,
            })

    # Add extra notable structures not yet used
    extra_entries = []
    for stem in EXTRA_NOTABLE:
        if stem not in used:
            cif_path = SAMPLES_DIR / f"{stem}.cif"
            if cif_path.exists():
                entry = build_entry(cif_path)
                if entry:
                    extra_entries.append(entry)
                    used.add(stem)
    if extra_entries:
        categories_out.append({
            "id": "notable",
            "label": "✨ Notable Structures",
            "entries": extra_entries,
        })

    total = sum(len(c["entries"]) for c in categories_out)
    print(f"\nTotal: {total} entries across {len(categories_out)} categories", flush=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({"categories": categories_out}, f, separators=(",", ":"))
    size_kb = OUT_FILE.stat().st_size / 1024
    print(f"Written to {OUT_FILE}  ({size_kb:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
