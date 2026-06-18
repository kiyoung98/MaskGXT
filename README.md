# MaskGXT

[![arXiv](https://img.shields.io/badge/arXiv-2606.22866-b31b1b.svg)](https://arxiv.org/abs/2606.22866)

Crystal structure prediction by masked generative modeling. Developed by an AI
co-scientist, [HACO](https://github.com/kiyoung98/HACO).

**Task.** Crystal structure prediction (CSP): given a chemical composition (the
atoms in the unit cell), predict its stable crystal structure — the lattice and
the atomic positions. A composition can have several stable structures
(polymorphs), so we measure both single-structure accuracy (match rate) and
polymorph coverage (METRe).

![overview](assets/overview.png)

A crystal is tokenized into one sequence (space group, lattice bins, per-atom
coordinate bins / Wyckoff / element). The transformer is trained to fill masked
tokens, and samples by iteratively unmasking from an all-masked sequence.

![architecture](assets/arch.png)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Datasets: `mp_20`, `mp_20_ps` (MP-20 polymorph split), `mpts_52`.

```bash
# 1. data (streamed from OMatG -> data/<dataset>_*.pt)
python prepare.py --dataset mp_20

# 2. train (checkpoint -> runs/<run_name>/best.pt; use --batch_size 128 for mpts_52)
python train.py --dataset mp_20 --run_name mp20

# 3. sample one CIF per test entry
python sample.py --dataset mp_20 --greedy --ckpt runs/mp20/best.pt   # match rate
python sample.py --dataset mp_20 --greedy --sg_stratify --ckpt runs/mp20/best.pt  # METRe

# 4. score (METRe match rate + cRMSE)
python evaluate.py --samples_dir runs/mp20/<decode tag>_samples --dataset mp_20
```

**Sampling flags** (independent): `--greedy` = MAP/argmax decoding;
`--sg_stratify` = assign distinct space groups across a composition's
generations for polymorph coverage. The Wyckoff/SG tables under `precompute/`
are committed; regenerate with `precompute_normalizer.py` / `precompute_wyckoff.py`.

## Citation

```bibtex
@misc{seong2026haco,
      title={Discovering Crystal Structure Prediction Algorithms with an AI Co-Scientist}, 
      author={Kiyoung Seong and Nayoung Kim and Sungsoo Ahn},
      year={2026},
      eprint={2606.22866},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.22866}, 
}
```
