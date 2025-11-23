import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from dr_sad.data.data_fetching import DOMAIN_SETTINGS
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = MAIN_DIR / "outputs"


def main(args) -> None:
    exp_name, exp_path = get_experiment_name(
        exp_name_arg=args.experiment_name,
        exp_config_dir=OUTPUT_DIR,
    )
    print(f"Collating results for experiment: {exp_name}")

    results = {}

    for result_file in sorted(exp_path.glob("domain_*/test_results.yaml")):
        domain_re = re.search(r"domain_(\d+)", result_file.parent.name)
        if domain_re is None:
            msg = f"Could not parse domain index from path: {result_file}"
            raise ValueError(msg)
        domain_idx = int(domain_re.group(1))
        with open(result_file) as f:
            domain_results = yaml.safe_load(f)

        results[domain_idx] = {
            "validation": domain_results["validation"]["test_accuracy"],
            "in_domain": domain_results["in_domain_test"]["test_accuracy"],
            "out_domain": domain_results["out_of_domain_test"]["test_accuracy"],
        }

    df = pd.DataFrame.from_dict(results, orient="index")

    if args.dataset_name is not None:
        if args.dataset_name not in DOMAIN_SETTINGS:
            msg = f"Unknown dataset_name: {args.dataset_name}"
            raise ValueError(msg)

        domain_names: dict[int, str] = {
            v: k
            for k, v in DOMAIN_SETTINGS[args.dataset_name]["domains_idx"].items()  # type: ignore[attr-defined]
        }

        df["domain"] = [domain_names[int(idx)] for idx in df.index]
    else:
        df["domain"] = df.index.astype(str)

    df = df[["domain", "validation", "in_domain", "out_domain"]]

    df.loc["mean"] = df.mean(numeric_only=True)
    df.loc["std"] = df.std(numeric_only=True)

    print("\nCollated Results:")
    print(df)

    df.to_csv(exp_path / "collated_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Location of the experiment directory to collate results from",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset for the domain names",
    )

    args = parser.parse_args()
    main(args)
