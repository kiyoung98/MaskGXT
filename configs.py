"""Dataset registry for MaskGXT.

A single training/sampling/eval codebase covers three crystal-structure
prediction benchmarks. The only things that differ per dataset are the maximum
number of atoms per unit cell (``max_n``), the number of coordinate and lattice
bins per axis (``k_bins``), and the OMatG source subdirectory used by
``prepare.py``. Everything else (model, loss, augmentation, sampler) is shared.

Dataset names follow the OMatG convention (``mp_20``, ``mp_20_ps``, ``mpts_52``)
and are used verbatim as: ``--dataset`` values, data file stems
(``<dataset>_{train,val,test}.pt``), the Wyckoff cache stem
(``wyckoff_cache_<dataset>.pt``), and the ``prepare.py`` download subdir.
"""
from __future__ import annotations

DATASETS = {
    "mp_20":    {"max_n": 20, "k_bins": 64},   # Materials Project, <=20 atoms/cell
    "mp_20_ps": {"max_n": 20, "k_bins": 64},   # MP-20 polymorph split
    # MPTS-52 is reported on the coarser grid: 37.21% three-seed match rate
    # against 36.33% at 64 bins.
    "mpts_52":  {"max_n": 52, "k_bins": 32},   # MPTS-52, chronological split, <=52 atoms/cell
}


def get_config(dataset: str) -> dict:
    """Return the per-dataset config dict, or raise on an unknown name."""
    if dataset not in DATASETS:
        raise SystemExit(
            f"unknown dataset {dataset!r}; choose one of {sorted(DATASETS)}"
        )
    return DATASETS[dataset]
