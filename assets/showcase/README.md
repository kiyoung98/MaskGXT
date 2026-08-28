# Showcase samples

Two MaskGXT generations picked for *large, high-symmetry* unit cells that the
model reproduces almost exactly. Selection: every generation was matched
index-aligned against its reference with
`StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10.0)`, then ranked by atom
count, reference space group, cell-volume error and RMS. RMS below is the
pymatgen normalized RMS, i.e. in units of `(V/N)^(1/3)`; the absolute value is
given alongside.

| file | reference | N | space group | RMS (norm / abs) | vol. error | generation |
|---|---|---|---|---|---|---|
| `SmCu3Mn4O12_mp-561515_MaskGXT.cif` | mp-561515 | 20 | Im-3 (#204) | 0.0053 / 0.011 A | +0.02% | MP-20 test, greedy decode |
| `Ba5Dy8Zn4O21_mp-18296_MaskGXT.cif` | mp-18296 | 38 | I4/m (#87) | 0.0292 / 0.071 A | +2.0% | MPTS-52, stochastic decode |

`*_reference.cif` is the corresponding ground-truth structure (symmetrized at
`symprec=0.1`) for side-by-side comparison.

Generated CIFs are written in P1 with every site listed, as the sampler emits
them. Running spglib on the coordinates recovers the reference space group:
Im-3 already at `symprec=0.05` for SmCu3Mn4O12, I4/m at `symprec=0.3` for
Ba5Dy8Zn4O21.

**SmCu3Mn4O12** is an AA'3B4O12 quadruple perovskite -- three cations on three
distinct sublattices. RCu3Mn4O12 compounds are colossal-magnetoresistance
ferrimagnets, and the isostructural CaCu3Ti4O12 is a giant-permittivity
capacitor material. **Ba5Ln8Zn4O21** lattices are studied as Eu3+/Dy3+ host
phosphors for white LEDs.

Provenance note: the MP-20 sample comes from the greedy decode used for the
one-to-one match rate (paper Table 1). The MPTS-52 sample comes from a
stochastic validation decode, not a greedy test decode.
