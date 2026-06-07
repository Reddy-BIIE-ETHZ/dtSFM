# Figures

Build scripts and source data for the main-text figures. Shared plotting style
lives in [`calm.figures.biie_style`](../src/calm/figures/biie_style.py).

| Figure | Content |
|--------|---------|
| Fig 1 | dtSFM architecture (frozen backbones → cross-attention encoder → 4 heads + decoder) |
| Fig 2 | Encoder retrieval dashboard, pool-512 R@K, ROC across clustering stringencies |
| Fig 3 | Off-target safety screening (Klaeger benchmark; ibrutinib case study, panel d) |
| Fig 4 | Library repurposing (NLRP3 / CD73 / STING1) with AF3 overlays (panel c) |
| Fig 5 | Generative design — cohort quality, orthogonality of dtSFM cosine vs AF3 |
| Fig 6 | 16-target structural-validation gallery (designs vs approved drug, anchor-overlaid; mean iPTM / PAE / Tanimoto per cell) |
| Fig 7 | Representative design close-ups (incl. PARP2 design d1062) |

## Scripts

All 42 figure-build scripts are in [`scripts/`](scripts/) — one or more per panel
(`build_fig<N>_*`, `make_fig*`, `render_*`). They share the plotting style in
[`calm.figures.biie_style`](../src/calm/figures/biie_style.py). Typical flow:
render structural overlays in PyMOL (`render_*` / `render_fig6_gallery.py`),
then run the matching `build_*` / `make_*` script with `PYTHONPATH=src`.

Figure panels are composited in PowerPoint/Illustrator from the script outputs;
panel letters and titles are added there.

> **Reproducibility note.** These scripts were authored against the manuscript
> repository layout and read their input tables from `audit/` (committed here) and
> from large data deposited to Zenodo (AF3 cofold CIFs/JSON, the full candidate
> pools, PyMOL render PNGs — not committed to git). Paths inside the scripts
> assume that original layout; adjust the `ROOT`/data paths to point at your local
> copy of the Zenodo deposit before re-running. The scripts are included for full
> transparency of how every published figure was generated.
