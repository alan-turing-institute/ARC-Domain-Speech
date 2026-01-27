import argparse
import pathlib

import pandas as pd
import yaml

from dr_sad.data.splitting import cross_validation_splitter


def main(args: argparse.Namespace):
    if not pathlib.Path(args.input_table).is_file():
        msg = f"Input table file '{args.input_table}' does not exist."
        raise FileNotFoundError(msg)

    if args.n_splits < 2:
        msg = "Number of splits must be at least 2."
        raise ValueError(msg)
    if args.n_splits > 26:
        msg = "Number of splits cannot exceed 26 (A-Z)."
        raise ValueError(msg)

    data = pd.read_csv(args.input_table, sep="\t", index_col=0, header=0)
    print(f"Loaded {len(data)} rows from {args.input_table}")

    if args.domain_column not in data.columns:
        msg = f"Domain column '{args.domain_column}' not found in data columns."
        raise ValueError(msg)
    domains = data[args.domain_column]

    data_folder = pathlib.Path(args.input_table).parent

    splits: list[dict[str, list[str]]] = cross_validation_splitter(
        domains, args.n_splits, args.val_ratio, args.random_seed
    )

    for i, split in enumerate(splits):
        output = {
            "train": split["train"],
            "val": split["val"],
            "test": split["test"],
        }
        identifier = chr(ord("A") + i)
        split_file = data_folder / f"split_{identifier}.yaml"
        with split_file.open("w") as f:
            yaml.safe_dump(output, f)

    print(f"Created {len(splits)} split files in {data_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train/val/test sets.")
    parser.add_argument("input_table", type=str, help="Path to the input TBL file.")
    parser.add_argument(
        "domain_column", type=str, help="Column name for domain identifiers."
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of cross-validation splits."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for the validation set.",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args()
    main(args)
