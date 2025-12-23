from pathlib import Path
from typing import TypedDict

import torch


def get_experiment_name(exp_name_arg: str, exp_config_dir: Path) -> tuple[str, Path]:
    """Get experiment name and path from argument."""
    p = Path(exp_name_arg)
    if p.exists():
        experiment_path = p
    elif (exp_config_dir / p).exists():
        experiment_path = exp_config_dir / p
    else:
        msg = f"Experiment config not found: {exp_name_arg}"
        raise FileNotFoundError(msg)
    # Determine experiment_name string
    try:
        # If experiment_path is inside EXP_CONFIG_DIR
        rel = experiment_path.relative_to(exp_config_dir)
        # Remove suffix (.yaml/.yml) and return the parent path + stem
        experiment_name = str(rel.with_suffix("")).replace("\\", "/")
    except ValueError:
        # Not inside EXP_CONFIG_DIR → just use the filename without suffix
        experiment_name = experiment_path.stem

    return experiment_name, experiment_path


class TrainingBatch(TypedDict):
    waveforms: torch.Tensor
    annotations: list[list[tuple[float, float]]]
    domains: list[int]
