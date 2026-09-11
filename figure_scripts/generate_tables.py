"""Generate combined DER tables comparing each experiment against its baseline.

Usage:
    python figure_scripts/generate_tables.py "outputs/paper_rerun_dihard5_*"
    python figure_scripts/generate_tables.py "outputs/paper_rerun_*"

For each lambda+model_type directory matching the pattern, the script loads
der_table.csv and combines it with the corresponding baseline_domain results
into a combined_der_table.csv (and .tex) saved in the same directory.
"""

import argparse
import re
import shutil
from glob import glob
from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Prefix segments to strip when resolving baseline directories.
# e.g. 'cyclicLR_hparam_sweep' -> strip '_hparam_sweep' -> 'cyclicLR'
PREFIX_STRIP_SEGMENTS = ["_hparam_sweep"]

# check more specific patterns first to avoid partial matches
MODEL_TYPE_PATTERNS = [
    ("adversarial_lstm", "_adversarial_lstm_lambda_"),
    ("adversarial", "_adversarial_lambda_"),
    ("irm", "_irm_lambda_"),
    ("vrex", "_vrex_lambda_"),
]

# (source, raw_column, display_name)
COLUMN_SPEC: list[tuple[str, str, str]] = [
    ("model", "validation", "ID Val"),
    ("model", "test", "ID Test"),
    ("model", "out_of_domain", "OOD"),
    ("baseline_domain", "out_of_domain", "OOD ERM"),
    ("baseline_single", "test_single", "Single"),
    ("baseline_all", "test_single", "All"),
]

PREFIX_DISPLAY_NAMES = {
    "paper_rerun_dihard5": "DIHARD5",
    "paper_rerun": "DIHARD",
    "cyclicLR": "DIHARD synthetic",
}

MODEL_DISPLAY_NAMES = {
    "adversarial": "Adversarial",
    "adversarial_lstm": "Adversarial LSTM",
    "irm": "IRM",
    "vrex": "V-REx",
}

# Columns each baseline contributes to the summary table
# Display names MUST match the display names in COLUMN_SPEC exactly.
BASELINE_COLUMN_MAP: dict[str, dict[str, str]] = {
    "baseline_domain": {
        "validation": "ID Val",
        "test": "ID Test",
        "out_of_domain": "OOD ERM",
    },
    "baseline_single": {
        "test_single": "Single",
    },
    "baseline_all": {
        "test_single": "All",
    },
}


def resolve_baseline_prefix(prefix: str) -> str:
    """Return the prefix used to locate baseline directories.

    For prefixes like 'cyclicLR_hparam_sweep', the baselines live under
    'cyclicLR_baseline_*', so we strip known intermediate segments.
    """
    for segment in PREFIX_STRIP_SEGMENTS:
        if segment in prefix:
            return prefix.replace(segment, "")
    return prefix


def parse_mean_value(value: str) -> float:
    """Extract the mean float from a '30.31(120)' string."""
    match = re.match(r"^([\d.]+)", str(value).strip())
    return float(match.group(1)) if match else float("nan")


def find_best_lambda(
    experiments: list[tuple[str, Path]],
    selection_col: str = "out_of_domain",
) -> tuple[str | None, Path | None, str | None]:
    """
    Return the (lambda_val, exp_path, stem) with the lowest mean DER on selection_col.
    """
    best_lambda, best_path, best_stem, best_der = None, None, None, float("inf")
    for lambda_val, exp_path in experiments:
        if float(lambda_val) == 0.0:
            continue
        der_csv = exp_path / "der_table.csv"
        if not der_csv.exists():
            continue
        df = pd.read_csv(der_csv, index_col=0)
        if "mean" not in df.index or selection_col not in df.columns:
            continue
        der = parse_mean_value(str(df.loc["mean", selection_col]))
        if der < best_der:
            best_der = der
            best_lambda = lambda_val
            best_path = exp_path
            model_type = get_prefix_and_model(exp_path.name)
            mt = model_type[1] if model_type else "model"
            best_stem = f"{mt}_lambda_{lambda_val.replace('.', 'p')}"
    return best_lambda, best_path, best_stem


def get_baseline_mean_row(der_csv: Path, col_map: dict[str, str]) -> dict[str, str]:
    """Read the mean row from a baseline der_table.csv, returning raw value strings.

    col_map maps raw column names to display names.
    """
    if not der_csv.exists():
        return {}
    df = pd.read_csv(der_csv, index_col=0)
    if "mean" not in df.index:
        return {}
    return {
        display: str(df.loc["mean", raw])
        for raw, display in col_map.items()
        if raw in df.columns
    }


def build_summary_df(
    prefix: str,
    model_experiments: dict[str, list[tuple[str, Path]]],
) -> pd.DataFrame:
    """Build a summary DataFrame with one row per model (best lambda) + baselines."""
    all_cols = [spec[2] for spec in COLUMN_SPEC]

    baseline_prefix = resolve_baseline_prefix(prefix)
    baseline_domain_csv = (
        OUTPUTS_DIR / f"{baseline_prefix}_baseline_domain" / "der_table.csv"
    )
    baseline_single_csv = (
        OUTPUTS_DIR / f"{baseline_prefix}_baseline_single" / "der_table.csv"
    )
    baseline_all_csv = OUTPUTS_DIR / f"{baseline_prefix}_baseline_all" / "der_table.csv"

    baseline_domain_vals = get_baseline_mean_row(
        baseline_domain_csv, BASELINE_COLUMN_MAP["baseline_domain"]
    )
    baseline_single_vals = get_baseline_mean_row(
        baseline_single_csv, BASELINE_COLUMN_MAP["baseline_single"]
    )
    baseline_all_vals = get_baseline_mean_row(
        baseline_all_csv, BASELINE_COLUMN_MAP["baseline_all"]
    )

    rows: dict[str, dict[str, str]] = {}

    # Baseline first — put OOD value in the "OOD" column, leave "OOD Baseline" blank
    baseline_row = dict.fromkeys(all_cols, "--")
    baseline_row.update(baseline_domain_vals)
    if "OOD ERM" in baseline_row and baseline_row["OOD ERM"] != "--":
        baseline_row["OOD"] = baseline_row["OOD ERM"]
        baseline_row["OOD ERM"] = "--"
    rows["ERM"] = baseline_row

    for model_type, experiments in model_experiments.items():
        lambda_val, best_path, _ = find_best_lambda(experiments)
        if best_path is None:
            continue
        df = pd.read_csv(best_path / "der_table.csv", index_col=0)
        row: dict[str, str] = dict.fromkeys(all_cols, "--")
        for source, raw_col, display_name in COLUMN_SPEC:
            if source == "model":
                if raw_col in df.columns and "mean" in df.index:
                    row[display_name] = str(df.loc["mean", raw_col])
            elif source == "baseline_domain":
                row[display_name] = baseline_domain_vals.get(display_name, "--")
            elif source == "baseline_single":
                row[display_name] = baseline_single_vals.get(display_name, "--")
            elif source == "baseline_all":
                row[display_name] = baseline_all_vals.get(display_name, "--")
        label = (
            f"{MODEL_DISPLAY_NAMES.get(model_type, model_type)} "
            f"($\\lambda$={lambda_val})"
        )
        rows[label] = row

    summary = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=all_cols)
    summary.index.name = "Model"
    return summary


def get_prefix_and_model(dirname: str) -> tuple[str, str] | None:
    """Extract the experiment prefix and model type from a directory name.

    For example, 'paper_rerun_dihard5_adversarial_lambda_0p0005' returns
    ('paper_rerun_dihard5', 'adversarial').

    Returns None if the directory doesn't match any known model type pattern.
    """
    for model_type, pattern in MODEL_TYPE_PATTERNS:
        idx = dirname.find(pattern)
        if idx != -1:
            return dirname[:idx], model_type
    return None


def get_lambda_value(dirname: str) -> str:
    """Extract the human-readable lambda value from a directory name."""
    match = re.search(r"_lambda_(.+)$", dirname)
    if match:
        return match.group(1).replace("p", ".")
    return "unknown"


def build_combined_df(
    model_df: pd.DataFrame,
    baseline_domain_df: pd.DataFrame,
    baseline_single_df: pd.DataFrame,
    baseline_all_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a flat pandas DataFrame with one column per entry in COLUMN_SPEC.

    Values are parsed to floats (stripping sample-count suffixes like '(120)').
    Missing source columns result in NaN.
    """
    sources = {
        "model": model_df,
        "baseline_domain": baseline_domain_df,
        "baseline_single": baseline_single_df,
        "baseline_all": baseline_all_df,
    }
    series: dict[str, pd.Series] = {}
    for source, raw_col, display_name in COLUMN_SPEC:
        df = sources[source]
        if raw_col in df.columns:
            series[display_name] = df[raw_col]
        else:
            series[display_name] = pd.Series("--", index=model_df.index)

    combined = pd.DataFrame(series)
    combined.index.name = "Domain"
    return combined


def format_domain_name(name: str) -> str:
    """Convert snake_case domain names to Title Case for LaTeX display."""
    return str(name).replace("_", " ").title()


def move_caption_to_bottom(latex: str) -> str:
    """Move \\caption and \\label lines to after \\end{tabular}."""
    caption_match = re.search(r"^\\caption\{.*\}\n", latex, re.MULTILINE)
    label_match = re.search(r"^\\label\{.*\}\n", latex, re.MULTILINE)
    if not caption_match:
        return latex
    caption_line = caption_match.group()
    label_line = label_match.group() if label_match else ""
    latex = re.sub(r"^\\caption\{.*\}\n", "", latex, flags=re.MULTILINE)
    latex = re.sub(r"^\\label\{.*\}\n", "", latex, flags=re.MULTILINE)
    return latex.replace(
        "\\end{table}\n", f"{caption_line}{label_line}\\end{{table}}\n"
    )


def df_to_latex(
    combined: pd.DataFrame, model_type: str, lambda_val: str, prefix: str = ""
) -> str:
    """Export the combined DataFrame to a LaTeX table string."""
    dataset = PREFIX_DISPLAY_NAMES.get(
        resolve_baseline_prefix(prefix), prefix.replace("_", " ")
    )
    caption = (
        f"DER (\\%) for "
        f"{MODEL_DISPLAY_NAMES.get(model_type, model_type.replace('_', ' '))}"
        f" on {dataset} with $\\lambda={lambda_val}$"
    )
    display = combined.copy()
    display.index = display.index.map(format_domain_name)
    display.index.name = None
    latex = display.to_latex(
        column_format=("l" + "|" + "c" * 3 + "|" + "c" + "|" + "c" * 2),
        na_rep="--",
        bold_rows=False,
        caption=caption,
        label=f"tab:der_{model_type}_lambda_{lambda_val.replace('.', 'p')}",
        position="ht",
    )
    # Replace the leading ' &' in the first header line with 'Domain &'
    latex = re.sub(r"(\\toprule\n) &", r"\1Domain &", latex)
    # Double rule after column headers (replace \midrule that follows the header)
    latex = re.sub(
        r"(Domain &.*?\\\\)\n\\midrule",
        r"\1\n\\midrule\\midrule",
        latex,
        count=1,
        flags=re.DOTALL,
    )
    # Insert \midrule before the Mean row
    latex = re.sub(r"(Mean &)", r"\\midrule\n\1", latex)
    return move_caption_to_bottom(latex)


def process_experiment(exp_path: Path) -> bool:
    """Process a single experiment directory.

    Writes the combined table into the experiment directory.
    Returns True if successfully processed, False otherwise.
    """
    der_csv = exp_path / "der_table.csv"
    if not der_csv.exists():
        return False

    result = get_prefix_and_model(exp_path.name)
    if result is None:
        return False

    prefix, model_type = result
    lambda_val = get_lambda_value(exp_path.name)

    baseline_prefix = resolve_baseline_prefix(prefix)
    baseline_domain_csv = (
        OUTPUTS_DIR / f"{baseline_prefix}_baseline_domain" / "der_table.csv"
    )
    baseline_single_csv = (
        OUTPUTS_DIR / f"{baseline_prefix}_baseline_single" / "der_table.csv"
    )
    baseline_all_csv = OUTPUTS_DIR / f"{baseline_prefix}_baseline_all" / "der_table.csv"

    if not baseline_domain_csv.exists():
        print(f"  Warning: No baseline_domain found for {exp_path.name}, skipping.")
        return False

    model_df = pd.read_csv(der_csv, index_col=0)
    baseline_domain_df = pd.read_csv(baseline_domain_csv, index_col=0)
    baseline_single_df = (
        pd.read_csv(baseline_single_csv, index_col=0)
        if baseline_single_csv.exists()
        else pd.DataFrame()
    )
    baseline_all_df = (
        pd.read_csv(baseline_all_csv, index_col=0)
        if baseline_all_csv.exists()
        else pd.DataFrame()
    )

    combined = build_combined_df(
        model_df, baseline_domain_df, baseline_single_df, baseline_all_df
    )

    stem = f"{model_type}_lambda_{lambda_val.replace('.', 'p')}"

    out_csv = exp_path / f"{stem}.csv"
    combined.to_csv(out_csv)

    latex = df_to_latex(
        combined,
        model_type,
        lambda_val,
        prefix=prefix,
    )
    out_tex = exp_path / f"{stem}.tex"
    out_tex.write_text(latex)

    print(f"  Saved: {out_csv.name}, {out_tex.name}")
    return True


def main(experiment_pattern: str) -> None:
    matching_dirs = sorted(glob(experiment_pattern))

    if not matching_dirs:
        print(f"No directories matched pattern: {experiment_pattern!r}")
        return

    print(f"Found {len(matching_dirs)} directories matching {experiment_pattern!r}")

    tables_dir = OUTPUTS_DIR / "experiment_tables"
    tables_dir.mkdir(exist_ok=True)

    # prefix -> model_type -> [(lambda_val, exp_path)]
    grouped: dict[str, dict[str, list[tuple[str, Path]]]] = {}

    processed = 0
    for exp_dir_str in matching_dirs:
        exp_path = Path(exp_dir_str)
        if not exp_path.is_dir():
            continue
        if process_experiment(exp_path):
            processed += 1
        result = get_prefix_and_model(exp_path.name)
        if result is not None:
            prefix, model_type = result
            lambda_val = get_lambda_value(exp_path.name)
            grouped.setdefault(prefix, {}).setdefault(model_type, []).append(
                (lambda_val, exp_path)
            )

    print(f"\nDone. Processed {processed} experiments.")

    # Generate one summary table per prefix
    print("\nGenerating summary tables...")
    for prefix, model_experiments in sorted(grouped.items()):
        summary = build_summary_df(prefix, model_experiments)
        if summary.empty:
            continue
        prefix_tables_dir = tables_dir / prefix
        prefix_tables_dir.mkdir(exist_ok=True)
        out_csv = prefix_tables_dir / f"{prefix}_summary.csv"
        summary.to_csv(out_csv)
        summary_display = summary.copy()
        summary_display.index.name = None
        exp_name = PREFIX_DISPLAY_NAMES.get(
            resolve_baseline_prefix(prefix), prefix.replace("_", " ")
        )
        latex = summary_display.to_latex(
            na_rep="--",
            caption=f"Best-lambda mean DER (\\%) summary for {exp_name}",
            label=f"tab:summary_{prefix}",
            position="ht",
            column_format=("l" + "|" + "c" * 3 + "|" + "c" + "|" + "c" * 2),
        )
        # Insert a \midrule after the baseline row (first data row)
        lines = latex.splitlines(keepends=True)
        data_rows_seen = 0
        result_lines = []
        for line in lines:
            result_lines.append(line)
            if line.strip().endswith(r"\\") and not line.strip().startswith(r"\\"):
                data_rows_seen += 1
                if data_rows_seen == 1:
                    result_lines.append("\\midrule\n")
        latex = "".join(result_lines)
        latex = move_caption_to_bottom(latex)
        out_tex = prefix_tables_dir / f"{prefix}_summary.tex"
        out_tex.write_text(latex)
        print(f"  {prefix}/{prefix}_summary.csv, {prefix}/{prefix}_summary.tex")

        # Copy only the best-lambda table for each model type into tables_dir
        for _, experiments in model_experiments.items():
            _, best_path, best_stem = find_best_lambda(experiments)
            if best_path is None or best_stem is None:
                continue
            for ext in (".csv", ".tex"):
                src = best_path / f"{best_stem}{ext}"
                if src.exists():
                    shutil.copy2(src, prefix_tables_dir / src.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate combined DER tables for each experiment vs its baseline_domain. "
            "Saves combined_der_table.csv and combined_der_table.tex into each "
            "matched experiment directory."
        )
    )
    parser.add_argument(
        "experiment_pattern",
        help=(
            "Glob pattern for experiment directories, e.g. "
            "'outputs/paper_rerun_dihard5_*' or 'outputs/paper_rerun_*'"
        ),
    )
    args = parser.parse_args()
    main(args.experiment_pattern)
