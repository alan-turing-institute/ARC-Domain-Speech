__all__ = ("stratified_splitter",)

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
        test_keys.extend([str(k) for k in chosen])

    remaining = domains.drop(index=test_keys)
    val_keys = []
    for domain in unique_domains:
        domain_keys = remaining[remaining == domain].index.to_list()
        k = int(np.ceil(len(domain_keys) * val_ratio))
        if k == 0:
            continue
        chosen = rng.choice(domain_keys, size=k, replace=False)
        val_keys.extend([str(k) for k in chosen])

    train_keys = list(remaining.drop(index=val_keys).index)
    train_keys = [str(k) for k in train_keys]

    train_keys.sort()
    val_keys.sort()
    test_keys.sort()

    return train_keys, val_keys, test_keys


def cross_validation_test_splitter(
    domains: pd.Series, n_splits: int, random_seed: int | np.random.Generator = 42
) -> list[list[str]]:
    """Create cross-validation splits for testing with stratification.

    Args:
        domains (pd.Series): A pandas Series where the index represents item identifiers
            and the values represent domain identifiers.
        n_splits (int): Number of cross-validation splits.
        random_seed (int | np.random.Generator, optional): Seed for the random number
            generator or a Generator instance. Defaults to 42.


    Returns:
        splits (list[list[str]]): A list containing n_splits lists of item identifiers
            for each test set.
    """
    # Make the random state
    if isinstance(random_seed, int):
        rng = np.random.default_rng(random_seed)
    else:
        rng = random_seed

    unique_domains = domains.unique()

    test_splits: list[list[str]] = [[] for _ in range(n_splits)]
    for i_d, domain in enumerate(unique_domains):
        # i_d is to ensure more even number of items per split
        domain_keys = domains[domains == domain].index.to_list()
        rng.shuffle(domain_keys)
        for i_k, key in enumerate(domain_keys):
            test_splits[(i_d + i_k) % n_splits].append(str(key))

    for split in test_splits:
        split.sort()

    return test_splits


def train_split_after_test(
    domains: pd.Series,
    test_keys: list[str],
    val_ratio: float = 0.1,
    random_seed: int | np.random.Generator = 42,
) -> tuple[list[str], list[str]]:
    """Create train and validation splits after a test split has been defined.

    Args:
        domains (pd.Series): A pandas Series where the index represents item identifiers
            and the values represent domain identifiers.
        test_keys (list[str]): List of item identifiers for the test set.
        val_ratio (float): Proportion of the remaining dataset to include in the
            validation split. Defaults to 0.1.
        random_seed (int | np.random.Generator, optional): Seed for the random number
            generator or a Generator instance. Defaults to 42.

    Returns:
        train_keys (list[str]): List of item identifiers for the training set.
        val_keys (list[str]): List of item identifiers for the validation set.
    """
    # Make the random state
    if isinstance(random_seed, int):
        rng = np.random.default_rng(random_seed)
    else:
        rng = random_seed

    remaining = domains.drop(index=test_keys)

    val_keys = []
    unique_domains = remaining.unique()
    for domain in unique_domains:
        domain_keys = remaining[remaining == domain].index.to_list()
        all_domain_keys = domains[domains == domain].index.to_list()
        k = int(np.ceil(len(all_domain_keys) * val_ratio))
        if k == 0:
            continue
        if k >= len(domain_keys):
            msg = (
                f"Not enough items in domain '{domain}' to allocate {k} "
                f"to validation set after test split. "
                f"The validation ratio may be too high."
            )
            raise ValueError(msg)

        chosen = rng.choice(domain_keys, size=k, replace=False)
        val_keys.extend([str(k) for k in chosen])

    train_keys = list(remaining.drop(index=val_keys).index)
    train_keys = [str(k) for k in train_keys]

    train_keys.sort()
    val_keys.sort()

    return train_keys, val_keys


def cross_validation_splitter(
    domains: pd.Series,
    n_splits: int,
    val_ratio: float = 0.1,
    random_seed: int | np.random.Generator = 42,
) -> list[dict[str, list[str]]]:
    """Create cross-validation splits with train/val sets for each fold.

    Args:
        domains (pd.Series): A pandas Series where the index represents item identifiers
            and the values represent domain identifiers.
        n_splits (int): Number of cross-validation splits.
        val_ratio (float): Proportion of the training dataset to include in the
            validation split. Defaults to 0.1.
        random_seed (int | np.random.Generator, optional): Seed for the random number
            generator or a Generator instance. Defaults to 42.

    Returns:
        splits (list[dict[str, list[str]]]): A list containing n_splits dictionaries,
            each with 'train', 'test', and 'val' keys mapping to lists of item
            identifiers.
    """

    if isinstance(random_seed, int):
        rng = np.random.default_rng(random_seed)
    else:
        rng = random_seed

    test_splits = cross_validation_test_splitter(domains, n_splits, random_seed=rng)

    splits = []
    for test_keys in test_splits:
        train_keys, val_keys = train_split_after_test(
            domains,
            test_keys,
            val_ratio=val_ratio,
            random_seed=rng,
        )
        split = {
            "train": train_keys,
            "val": val_keys,
            "test": test_keys,
        }
        splits.append(split)

    return splits
