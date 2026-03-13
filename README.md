# Single-Target Ligand Screening Research Prototype

Ligand-only ML pipeline for activity prediction and virtual screening.

> Scope: representation learning + regression + screening ranking.
> Non-scope: molecule generation/decoding, docking execution.

---

## Project Structure

```text
agent_test/
  configs/
    default.yaml                  # base config
    experiments/                  # 6 experiment configs (mlp|gnn|sme × scaffold|random)
  data/
    BACE1/                        # BACE1 activity dataset (ChEMBL)
  src/
    data/       loader, cleaner, splitter
    features/   fingerprints, descriptors, graph (SME)
    models/     mlp, gnn (AttentiveFP), sme_rgcn, ensemble
    training/   trainer, scheduler
    evaluation/ metrics, plots
    screening/  ranker, rerank, latent_opt
    utils/      config, io, seed
  scripts/
    preprocess.py
    train.py
    evaluate.py
    screen.py
    optimize_latent.py
    run_all_experiments.sh        # 6개 실험 일괄 실행
    submit.sh                     # SLURM SBATCH 스크립트
  outputs/runs/<exp_name>/        # 실험별 결과 (gitignore)
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate ligand_screen
```

---

## Data Format

`data/` 하위 CSV는 최소한 다음 컬럼 필요:

| 컬럼 | 설명 |
|------|------|
| `smiles` | SMILES string |
| `pIC50` | 활성값 (regression target) |

`configs/default.yaml`의 `data.smiles_col`, `data.activity_col`로 컬럼명 변경 가능.

현재 기본 데이터셋: **BACE1** (`data/BACE1/bace1_clean_pic50.csv`, 8,184개 분자)

---

## Models

| `model.type` | `features.type` | 설명 |
|---|---|---|
| `mlp` | `fingerprint` | Morgan ECFP + MLP |
| `gnn` | `graph` | AttentiveFP (PyG) |
| `sme_rgcn` | `sme_graph` | SME RGCN (Wu et al. 2023, BRICS substructure masking) |

---

## End-to-End Run (단일 실험)

```bash
export PYTHONPATH=.

# 1. Preprocess + split
python scripts/preprocess.py --config configs/default.yaml

# 2. Train
python scripts/train.py --config configs/default.yaml

# 3. Evaluate
python scripts/evaluate.py --config configs/default.yaml

# 4. Screen unlabeled library
python scripts/screen.py --config configs/default.yaml --screen_csv data/screen_library.csv
```

---

## 6-Experiment Benchmark (mlp|gnn|sme × scaffold|random)

```bash
# 로컬
bash scripts/run_all_experiments.sh

# SLURM 서버
sbatch scripts/submit.sh
```

각 실험 결과는 `outputs/runs/<exp_name>/`에 독립 저장됨.

---

## Outputs

`outputs/runs/<exp_name>/` 구조:

```text
cleaned_dataset.csv
config_resolved.yaml
splits/          train_idx.npy, val_idx.npy, test_idx.npy
checkpoints/     best.pt, epoch_10.pt, ...
predictions/     train_preds.csv, val_preds.csv, test_preds.csv
latents_train.npy / latents_val.npy / latents_test.npy
history.json
```

---

## Reproducibility

- `training.seed: 42` (config에서 변경 가능)
- `src/utils/seed.py`: random / numpy / torch / cuda 모두 고정
- config + split indices + checkpoints 저장으로 exact rerun 가능

---

## Notes

- `src/screening/rerank.py`의 `rerank_hook(scores, metadata)`: 현재 identity, 향후 docking score 연결 지점
- `scripts/optimize_latent.py`: latent-space 최적화 starter workflow
