# ARC-Domain-Speech

[![Actions Status][actions-badge]][actions-link]

ARC project on domain-robust speech activity detection (DRSAD).

## Overview

Speech activity detection models are typically trained on data from a single acoustic
domain (e.g. telephone calls, broadcast news, or meeting recordings). Performance often
degrades significantly when such models are deployed in conditions that differ from
training. This project investigates domain robustness techniques that improve
performance across diverse acoustic environments.

The package implements a PyTorch Lightning training pipeline built around the
[PyanNet](https://github.com/pyannote/pyannote-audio) architecture and several
domain adaptation methods — adversarial training, Invariant Risk Minimisation (IRM),
and Variance Risk Extrapolation (V-REx) — predocminantly evaluated on configurations of the DIHARD dataset.

## Architecture

### Baseline: PyanNet

The baseline model processes raw waveforms through three stages:

```
Waveforms -> SincNet (learnable sinc filterbanks -> Conv1d blocks -> max-pool)
          -> BiLSTM  (2 layers, bidirectional, 128 -> 256 units)
          -> Linear  (2 × 128 feed-forward)
          -> Classifier (sigmoid -> frame-level speech probability)
```

### Domain Adaptation Methods

| Model key | Method | Summary |
|---|---|---|
| `default_pyannet` | Baseline | PyanNet with no domain adaptation, trains via ERM $L_{\text{ERM}}$ |
| `adversarial_net` | Adversarial (at Linear) | Gradient reversal after the linear head forces domain-invariant features; loss weighted by $\lambda$ |
| `adversarial_lstm` | Adversarial (at LSTM) | Adversarial head applied after the LSTM layer ,also loss weighted by $\lambda$ |
| `irm_model` | IRMv1 | Finds a feature representation with equal risk across domains; supports $\lambda$ scheduling |
| `vrex_model` | V-REx | Penalises high variance of per-domain losses: $L = L_\text{ERM} + \lambda \cdot \text{Var}(L_e)$ |

## Datasets

| Dataset | Split role | Notes |
|---|---|---|
| [DIHARD](https://catalog.ldc.upenn.edu/LDC2022S14) | Train / val / test | Primary benchmark; contains recordings from multiple domains (meetings, audiobooks, restaurant, etc.) |
| Synthetic |  Train / val / test | Programmatically generated data from DIHARD for increased difficulty |
| MUSAN | Noise augmentation | Background noise applied at configurable SNR during training |
| CallHome | Supplementary | Telephone-speech recordings |

Datasets are **not** included in this repository. Each dataset must be obtained
separately and placed under the `data/<dataset-name>/` directory. See the data
configuration files in `configs/data/` for the expected directory names.

Notes on dataset preparation can be found in `data/README.md`.

## Installation

**Python 3.10 or later is required.**

From source:

```bash
git clone https://github.com/alan-turing-institute/ARC-Domain-Speech
cd ARC-Domain-Speech
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

With `uv`:

```bash
git clone https://github.com/alan-turing-institute/ARC-Domain-Speech
cd ARC-Domain-Speech
uv venv
uv sync
source .venv/bin/activate
```

## Configuration System

Experiments are described by three layers of YAML configuration files:

```
configs/
├── experiment/   <- top-level config; references the three configs below
├── data/         <- dataset, splits, domain type, noise augmentation
├── model/        <- model class and hyperparameters (e.g. lambda_scheduling)
└── training/     <- max epochs, batch size, learning rate, early stopping
```

An experiment config combines the individual configs:

```yaml
# configs/experiment/dihard_adversarial_0p02_domain_30.yaml
data_config:     dihard_domain.yaml
training_config: bask_default.yaml
model_config:    adversarial_lambda0p02.yaml
random_seed:     42
time_slice:      30.0   # audio chunk duration (seconds) during training
batch_multiplier: 8
```

A model config specifies the model class and any hyperparameters:

```yaml
# configs/model/adversarial_lambda0p02.yaml
model_name: adversarial_net
domain_loss_weight: 0.02
```

A data config controls how domain splits are handled:

```yaml
# configs/data/dihard_domain.yaml
name: dihard
split_names:
  - split_A.yaml
  - split_B.yaml
  - split_C.yaml
  - split_D.yaml
  - split_E.yaml
domain_type: exclude_one   # or: all | single_domain
```

`domain_type` controls the training strategy:

| Value | Behaviour |
|---|---|
| `all` | Train on all domains combined |
| `single_domain` | Train on one domain only (requires `--domain`) |
| `exclude_one` | Train on all domains except one held-out domain (requires `--domain`) |

## Usage

### Training

```bash
python scripts/train.py <experiment_name> <split_idx> [--domain DOMAIN]
```

| Argument | Description |
|---|---|
| `experiment_name` | Name of the experiment config (with `.yaml` extension), e.g. `dihard_domain_30.yaml` |
| `split_idx` | Index into the `split_names` list in the data config (0–4 for 5-fold cross-validation) |
| `--domain` | Integer domain index to use/exclude; required when `domain_type` is `exclude_one` or `single_domain` |

**Example — leave-one-domain-out with adversarial training:**

```bash
python scripts/train.py dihard_adversarial_0p02_domain_30 0 --domain 2
```

Outputs are written to `outputs/<experiment_name>/<split_name>/[domain_<N>/]`:

```
outputs/
└── dihard_adversarial_0p02_domain_30/
    └── split_A/
        └── domain_2/
            ├── final_checkpoint.ckpt
            ├── trained_model_weights.safetensors
            ├── test_results.yaml
            └── version_0/
                └── metrics.csv
```

### Prediction

After training, generate frame-level speech probability predictions:

```bash
python scripts/predict.py <experiment_config> <split_idx> [--domain DOMAIN]
```

Arguments match those of `train.py`. Predictions are saved as
[safetensors](https://github.com/huggingface/safetensors) files alongside model
metadata:

```
outputs/<experiment_name>/<split_name>/[domain_<N>/]
├── model_metadata.yaml          <- sample rate, frame hop, receptive field, etc.
└── saved_predictions/
    ├── validation.safetensors
    ├── test.safetensors            ('exclude_one' and 'all' modes only)
    └── out_of_domain.safetensors   ('exclude_one' mode only)
    └── test_single.safetensors     ('single_domain' mode only)
```

### Analysis

Run the following `frame_analysis.py` and `segment_analysis.py` for each split/domain combination, then collate once across the whole experiment.

**1. Frame-level analysis**

```bash
python scripts/frame_analysis.py <experiment_config> <split_idx> [--domain N] [--use-collar]
```

Computes frame-level metrics (DER, F1, precision, recall) from the saved predictions.
Output: `outputs/<experiment_name>/<split_name>/[domain_<N>]/frame_metrics.yaml`

`--use-collar` adds a 0.25 s collar around segment boundaries before scoring (default: off).

**2. Segment analysis**

```bash
python scripts/segment_analysis.py <experiment_config> <split_idx> [--domain N] [--no-inverse-weightings]
```

Optimises post-processing parameters (speech threshold, gap fill, minimum speech/silence durations) on the validation set via differential evolution, then evaluates all prediction sets.
Output: `outputs/<experiment_name>/<split_name>/[domain_<N>]/segment_analysis.yaml`

`--no-inverse-weightings` disables domain-frequency-based sample weighting when computing mean metrics (default: inverse weighting enabled).

**3. Collate metrics** *(run once after all splits and domains are complete)*

```bash
python scripts/collate_all_metrics.py <experiment_config>
```

Merges `frame_metrics.yaml` and `segment_analysis.yaml` across all domains and splits, producing summary tables at the experiment level.
Outputs: `all_metrics.yaml` per split, `all_metrics.yaml` at the experiment root, `der_table.csv`, `frame_f1_table.csv`.

### Hyperparameter Sweep Analysis

After running sweeps across multiple $\lambda$ values, plot the results:

```bash
python scripts/hparam_sweep.py <filepath_pattern>
```

`filepath_pattern` is matched against `outputs/` to locate sweep experiment
directories (e.g. `hparam_sweep` matches directories like
`hparam_sweep_adversarial_lambda_0p02`). The script produces a DER vs $\lambda$
plot saved to `outputs/figures/`.

It requires files be named with a common prefix (e.g. `hparam_sweep`), a set of model identifiers (eg. `adversarial`, `IRM`, etc.. ) and a lambda value (eg. `_lambda_0p02` -> $\lambda$ = 0.02). Baseline runs (`domain`, `single`, and `all`) should be named with the prefix and then `_baseline_[RUN_TYPE]` (eg. `hparam_sweep_baseline_single`).

## Project Structure

```
ARC-Domain-Speech/
├── configs/
│   ├── data/          # dataset and domain-split configs
│   ├── experiment/    # top-level experiment configs
│   ├── model/         # model + hyperparameter configs
│   └── training/      # trainer, LR scheduler, early stopping configs
├── data/              # datasets (not included)
├── outputs/           # training artefacts and predictions (git-ignored)
├── scripts/
│   ├── train.py                  # train a model
│   ├── predict.py                # generate predictions from a trained model
│   ├── frame_analysis.py         # compute frame-level metrics
│   ├── segment_analysis.py       # optimise post-processing and compute segment-level metrics
│   ├── collate_all_metrics.py    # aggregate metrics across all domains and splits
│   ├── hparam_sweep.py           # plot λ-sweep results
│   ├── data_splitting.py         # create train/val/test splits
│   └── ...
├── src/dr_sad/
│   ├── pyannet/       # SincNet, BiLSTM, PyanNet model
│   ├── models/        # AdversarialNet, IRMv1Model, VRExModel, AdversarialLSTM
│   ├── data/          # dataset loading, dataloaders, noise augmentation
│   ├── training.py    # DrSadTrainer, MODEL_DICT, create_model()
│   ├── predicting.py  # inference pipeline
│   ├── analysis.py    # F1, precision, recall, ROC-AUC
│   └── ...
└── tests/             # pytest test suite
```

## License

Distributed under the terms of the [MIT license](LICENSE).

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/ARC-Domain-Speech/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/ARC-Domain-Speech/actions
<!-- prettier-ignore-end -->
