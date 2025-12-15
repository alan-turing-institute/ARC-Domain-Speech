from pathlib import Path
from typing import Any


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


def flatten_dict(dict_to_flatten: dict[str, Any], sep: str = "_") -> dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        dict_to_flatten (dict): The dictionary to flatten.
        sep (str, optional): The separator between keys. Defaults to "_".

    Returns:
        dict[str, int | float | str]: The flattened dictionary.
    """
    new_dict: dict[str, Any] = {}
    for k, v in dict_to_flatten.items():
        if isinstance(v, dict):
            flat_dict = flatten_dict(v, sep=sep)
            for fk, fv in flat_dict.items():
                if not isinstance(fk, str):
                    msg = f"Expected string key, got {type(fk)}"  # type: ignore[unreachable]
                    raise TypeError(msg)
                new_dict[f"{k}{sep}{fk}"] = fv
        else:
            if not isinstance(k, str):
                msg = f"Expected string key, got {type(k)}"  # type: ignore[unreachable]
                raise TypeError(msg)
            new_dict[k] = v

    return new_dict
