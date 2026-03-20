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
    experiments/                  # baseline + diffusion configs
                                  # mlp|gnn|sme × scaffold|random (baseline)
                                  # gnn_unet_random, gnn_unet_vqc_random (diffusion)
                                  # sme_unet_random, sme_unet_vqc_random (diffusion)
  data/
    BACE1/                        # BACE1 activity dataset (ChEMBL)
    bace1_dataset.py              # BACE1 dataset utilities
  src/
    data/       loader, cleaner, splitter
    features/   fingerprints, descriptors, graph (SME)
    models/     mlp, gnn (AttentiveFP), sme_rgcn, ensemble
                diffusion (denoisers: mlp, unet, unet_vqc)
                vqc_module.py (quantum bottleneck circuits)
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
    train_diffusion.py            # diffusion-based latent optimization
    visualize_vqc.py              # VQC circuit visualization
    run_all_experiments.sh        # baseline + diffusion experiments
    submit.sh                     # SLURM SBATCH 스크립트
  outputs/runs/<exp_name>/        # 실험별 결과 (gitignore)
             /diffusion/          # diffusion outputs
                /T{T}_ep{epochs}/
                  denoiser_cfg.pt (best)
                  denoiser_cfg_final.pt
                  HybridUNetDenoiser_arch.html
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

### Baseline Models

| `model.type` | `features.type` | 설명 |
|---|---|---|
| `mlp` | `fingerprint` | Morgan ECFP + MLP |
| `gnn` | `graph` | AttentiveFP (PyG) |
| `sme_rgcn` | `sme_graph` | SME RGCN (Wu et al. 2023, BRICS substructure masking) |

### Diffusion-based Latent Optimization

Denoiser architecture selectable via `config.diffusion.denoiser_type`:

| Denoiser Type | Class | 설명 |
|---|---|---|
| `mlp` | ConditionalDenoisingMLP | Flat 6-layer MLP (original) |
| `unet` | HybridUNetDenoiser | U-Net encoder-decoder with classical bottleneck |
| `unet_vqc` | HybridUNetDenoiser | U-Net with quantum VQC bottleneck (ring entanglement, 5 qubits) |

**HybridUNetDenoiser Architecture** (`src/models/vqc_diffusion.py`):
- **Encoder**: 256 → 128 → 64 → 32 (stride=2, skip connections)
- **Bottleneck**: AmplitudeEmbedding → ring entanglement circuit → PauliZ measurements → Linear projection
- **Decoder**: 32 → 64 → 128 → 256 (upsample, cat+skip, residual blocks)
- **VQC Params**: `n_layers: 2` (circuit depth), `bottleneck_dim: 32` (5 qubits)
- **Config**: `unet_dims: [128, 64, 32]` (encoder/decoder dimensions)

---

## End-to-End Run (단일 실험)

### Baseline Model Training

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

### Diffusion-based Latent Optimization

```bash
export PYTHONPATH=.

# 1. Train baseline model (from above)
# 2. Run diffusion latent optimization
python scripts/train_diffusion.py \
  --config configs/experiments/gnn_unet_vqc_random.yaml \
  --baseline_run outputs/runs/gnn_random/ \
  --T 1000 \
  --denoiser_type unet_vqc

# Available denoiser types: mlp, unet, unet_vqc
# Output: outputs/runs/<exp_name>/diffusion/T{T}_ep{epochs}/
```

### VQC Circuit Visualization

```bash
python scripts/visualize_vqc.py \
  --bottleneck_dim 32 \
  --n_layers 2 \
  --output outputs/HybridUNetDenoiser_arch.html
```

---

## Experiment Configs

### Baseline Experiments (6 configs)

- `mlp_scaffold_split.yaml`, `mlp_random_split.yaml`
- `gnn_scaffold_split.yaml`, `gnn_random_split.yaml`
- `sme_rgcn_scaffold_split.yaml`, `sme_rgcn_random_split.yaml`

### Diffusion Experiments (4 configs)

- `gnn_unet_random.yaml` (denoiser_type: `unet`)
- `gnn_unet_vqc_random.yaml` (denoiser_type: `unet_vqc`)
- `sme_unet_random.yaml` (denoiser_type: `unet`)
- `sme_unet_vqc_random.yaml` (denoiser_type: `unet_vqc`)

### Config Diffusion Section

All configs include:

```yaml
diffusion:
  denoiser_type: "mlp"          # "mlp" | "unet" | "unet_vqc"
  unet_dims: [128, 64, 32]      # encoder/decoder dimensions
  n_layers: 2                   # VQC circuit depth
```

### Benchmark Runs

```bash
# 로컬 (baseline + diffusion)
bash scripts/run_all_experiments.sh

# SLURM 서버
sbatch scripts/submit.sh
```

각 실험 결과는 `outputs/runs/<exp_name>/`에 독립 저장됨.
Diffusion outputs: `outputs/runs/<exp_name>/diffusion/T{T}_ep{epochs}/`

---

## Outputs

### Baseline Model Outputs

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

### Diffusion Outputs

`outputs/runs/<exp_name>/diffusion/T{T}_ep{epochs}/` 구조:

```text
denoiser_cfg.pt           # best-loss checkpoint
denoiser_cfg_final.pt     # final-epoch checkpoint
HybridUNetDenoiser_arch.html  # architecture diagram
diffusion_history.json    # training history
```

Checkpoint Path Format:
- **Best**: `diffusion/T1000_ep100/denoiser_cfg.pt`
- **Final**: `diffusion/T1000_ep100/denoiser_cfg_final.pt`
- **Epoch count** included in folder name for easy tracking

---

## Reproducibility

- `training.seed: 42` (config에서 변경 가능)
- `src/utils/seed.py`: random / numpy / torch / cuda 모두 고정
- config + split indices + checkpoints 저장으로 exact rerun 가능

---

## Configuration Details

### Denoiser Type Selection

```bash
# Command line override
python scripts/train_diffusion.py --config config.yaml --denoiser_type unet_vqc

# Config file
diffusion:
  denoiser_type: "unet_vqc"
  unet_dims: [128, 64, 32]
  n_layers: 2
```

### Configurable Hyperparameters

- `latent_dim`: Output dimension (default: 128, configurable per experiment)
- `unet_dims`: U-Net layer dimensions (default: [128, 64, 32])
- `n_layers`: VQC circuit depth (default: 2)
- `denoiser_type`: mlp | unet | unet_vqc

---

## Notes

- `src/screening/rerank.py`의 `rerank_hook(scores, metadata)`: 현재 identity, 향후 docking score 연결 지점
- `scripts/optimize_latent.py`: baseline latent-space 최적화 starter workflow
- `scripts/train_diffusion.py`: diffusion-based latent optimization with denoiser selection
- `src/models/vqc_module.py`: quantum circuit components (AmplitudeEmbedding, ring entanglement)
- VQC bottleneck uses 5 qubits for `bottleneck_dim=32` (fixed mapping)
