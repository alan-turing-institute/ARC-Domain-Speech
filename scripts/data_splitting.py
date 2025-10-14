import argparse
import pathlib

import pandas as pd
import yaml

from dr_sad.data.splitting import stratified_splitter


def main(args: argparse.Namespace):
    data = pd.read_csv(args.input_table, sep="\t", index_col=0, header=0)
    print(f"Loaded {len(data)} rows from {args.input_table}")
    domains = data[args.domain_column]

    train_keys, val_keys, test_keys = stratified_splitter(
        domains, args.test_ratio, args.val_ratio, args.random_seed
    )

    output = {
        "train": train_keys,
        "val": val_keys,
        "test": test_keys,
    }

    with open(pathlib.Path(args.output_yaml), "w") as f:
        yaml.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train/val/test sets.")
    parser.add_argument("input_table", type=str, help="Path to the input TBL file.")
    parser.add_argument(
        "domain_column", type=str, help="Column name for domain identifiers."
    )
    parser.add_argument("output_yaml", type=str, help="Path to the output YAML file.")
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Proportion of data for the test set.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Proportion of data for the validation set.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args()
    main(args)
