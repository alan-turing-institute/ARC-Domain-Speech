# Figure Scripts

Scripts used to generate the figures and tables for our report.

## Note:
- These reference config/experiment names from our own runs (e.g. files under `configs/experiment`, and hardcoded names/prefixes like `paper_rerun_*`, `hparam_sweep_*`, `baseline_*`), so they won't work out of the box on other configs.
- You may need to do some tinkering (paths, config names, domain settings, etc.) to adapt them to new experiments.

## Scripts

### `precision_recall_curve.py`
Generates per-domain precision-recall curves for a single experiment, from the saved
predictions of each trained model. Results are cached to
`outputs/<exp_name>/precision_recall_curve_data.yaml` so the curves can be replotted
without re-running the evaluation.

```bash
python figure_scripts/precision_recall_curve.py <experiment_config> [--use-existing-results]
```

### `baseline_pr_curves.py`
Plots the baseline PR curves for the three training setups (trained on all domains,
all-except-domain, and single-domain) on one grid of per-domain axes. Reads the cached
`precision_recall_curve_data.yaml` produced by `precision_recall_curve.py`, so run that
for each of the three experiments first.

```bash
python figure_scripts/baseline_pr_curves.py --all <config> --single <config> --domain <config>
```

### `generate_tables.py`
Builds the combined DER tables (CSV + LaTeX) comparing each experiment against its
baselines. Takes a glob over experiment output directories, picks the best lambda per
model type by OOD DER, and writes `combined_der_table.csv`/`.tex` into the matched
directories. Model types (`irm`, `vrex`, `adversarial`, `adversarial_lstm`) and dataset
display names are hardcoded at the top of the file.

```bash
python figure_scripts/generate_tables.py "outputs/paper_rerun_*"
```

### `hparam_sweep.py`
Plots DER against the regularisation strength (lambda) for the adversarial sweeps, one
panel per model variant, with the ERM baseline marked on the x-axis. The set of sweep
directory names it looks for is hardcoded in the `models` dict in `main`.

```bash
python figure_scripts/hparam_sweep.py <filepath_pattern>
```

### `get_speech_ratios.py`
Computes the per-domain speech ratio (percentage of audio that is speech) for DIHARD
and writes `speech_ratios.tsv` and `speech_ratios.tex` into `data/dihard`. Expects the
dataset to be present locally; takes no arguments.

```bash
python figure_scripts/get_speech_ratios.py
```
