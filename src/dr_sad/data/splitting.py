import numpy as np
import pandas as pd


def stratified_splitter(
    domains: pd.Series, test_ratio: float, val_ratio: float, random_seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """Split a table into train, validation, and test sets with stratification.

    Args:
        domains (pd.Series): A pandas Series where the index represents item identifiers
            and the values represent domain identifiers.
        test_ratio (float): Proportion of the dataset to include in the test split.
        val_ratio (float): Proportion of the dataset to include in the validation split.
        random_seed (int, optional): Seed for the random number generator.
            Defaults to 42

    Returns:
        train_keys (list[str]): List of item identifiers for the training set.
        val_keys (list[str]): List of item identifiers for the validation set.
        test_keys (list[str]): List of item identifiers for the test set.
    """

    # Make the random state
    rng = np.random.default_rng(random_seed)

    unique_domains = domains.unique()

    test_keys = []
    for domain in unique_domains:
        domain_keys = domains[domains == domain].index.to_list()
        k = int(np.ceil(len(domain_keys) * test_ratio))
        if k == 0:
            continue
        chosen = rng.choice(domain_keys, size=k, replace=False)
        test_keys.extend(chosen.tolist())

    remaining = domains.drop(index=test_keys)
    val_keys = []
    for domain in unique_domains:
        domain_keys = remaining[remaining == domain].index.to_list()
        k = int(np.ceil(len(domain_keys) * val_ratio))
        if k == 0:
            continue
        chosen = rng.choice(domain_keys, size=k, replace=False)
        val_keys.extend(chosen.tolist())

    train_keys = list(remaining.drop(index=val_keys).index)

    return train_keys, val_keys, test_keys
