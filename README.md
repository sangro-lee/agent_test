# Single-Target Ligand Screening Research Prototype

Ligand-only ML pipeline for activity prediction and virtual screening.

> Scope: representation learning + regression + screening ranking.  
> Non-scope in this version: molecule generation/decoding, docking execution.

## Project Structure

```text
project/
  configs/
  data/
  src/
    data/
    features/
    models/
    training/
    evaluation/
    screening/
    utils/
  scripts/
  outputs/
  README.md
```

## Required Libraries

- numpy
- pandas
- scikit-learn
- matplotlib
- pyyaml
- torch
- rdkit

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Input CSV Requirements

Activity dataset must include at least:
- `smiles` (or your configured `smiles_col`)
- `activity` (or your configured `activity_col`)

Optional:
- `unit` column (`nM`, `uM`, etc.) when converting IC50/Ki/Kd to pIC50.

Screening library CSV must include:
- `smiles` (or configured smiles column)

## End-to-End Run

1) Preprocess + split
```bash
python scripts/preprocess.py --config configs/default.yaml
```

2) Train model + export predictions/latents
```bash
python scripts/train.py --config configs/default.yaml
```

3) Evaluate (RMSE/MAE/R2/Pearson + plots)
```bash
python scripts/evaluate.py --config configs/default.yaml
```

4) Screen unlabeled library
```bash
python scripts/screen.py --config configs/default.yaml --screen_csv data/screen_library.csv
```

## Outputs

Under `outputs/runs/YYYYMMDD_HHMMSS/` by default (`output.run_root: auto`):
- `cleaned_dataset.csv`
- `splits/{train,val,test}_idx.npy`
- `models/best_val_loss_epoch=*.pt`
- `models/final_model.pt`
- `models/checkpoints/epoch_*.pt` (if `checkpoint_every > 0`)
- `pred_{train,val,test}.csv`
- `latents/{train,val,test}_latents.npy/.pt`
- `metrics.json`
- `plots/*`
- `screening/ranked_all.csv`, `screening/top_k.csv`, `screening/screen_latents.npy/.pt`

## Design Notes for Future Docking Reranking

- `src/screening/rerank.py` exposes `rerank_hook(scores, metadata)`.
- Current behavior is identity mapping.
- Future docking/ensemble scores can be merged there without refactoring training pipeline.

## Latent-Space Search Note

- Current `scripts/screen.py` only reranks an existing molecule library.
- Recommended first step for latent-space optimization is:
  optimize latent vectors directly with the trained regression head while
  penalizing distance from the train latent manifold, then retrieve nearest
  known molecules for interpretation.
- A starter workflow is provided in `scripts/optimize_latent.py`.

## Reproducibility

- Deterministic seed control (`src/utils/seed.py`)
- Config-driven runs (`configs/default.yaml`)
- Saved indices/checkpoints/artifacts for exact reruns
