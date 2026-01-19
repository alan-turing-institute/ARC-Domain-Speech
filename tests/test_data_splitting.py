import numpy as np
import pandas as pd

from dr_sad.data import splitting


class TestStratifiedSplitter:
    def expected_counts_by_domain(
        self, domains: pd.Series, test_ratio: float, val_ratio: float
    ) -> dict[str, dict[str, int]]:
        expected = {}
        for domain in domains.unique():
            n = int((domains == domain).sum())
            test_k = int(np.ceil(n * test_ratio))
            remaining = n - test_k
            val_k = int(np.ceil(remaining * val_ratio))
            expected[domain] = {
                "n": n,
                "test": test_k,
                "val": val_k,
                "train": remaining - val_k,
            }
        return expected

    def test_stratified_counts_and_partitioning(self):
        indices = [f"i{j}" for j in range(1, 13)]
        labels = ["A", "A", "A", "A", "B", "B", "B", "C", "C", "D", "D", "D"]
        domains = pd.Series(labels, index=indices)

        test_ratio = 0.25
        val_ratio = 0.3

        train_keys, val_keys, test_keys = splitting.stratified_splitter(
            domains, test_ratio, val_ratio, random_seed=123
        )

        # types
        assert isinstance(train_keys, list)
        assert isinstance(val_keys, list)
        assert isinstance(test_keys, list)

        assert all(isinstance(k, str) for k in train_keys)
        assert all(isinstance(k, str) for k in val_keys)
        assert all(isinstance(k, str) for k in test_keys)

        # disjointness and coverage
        all_keys = set(train_keys) | set(val_keys) | set(test_keys)
        assert all_keys == set(indices)
        assert set(train_keys).isdisjoint(set(val_keys))
        assert set(train_keys).isdisjoint(set(test_keys))
        assert set(val_keys).isdisjoint(set(test_keys))

        # per-domain counts match the ceil-based logic used by the function
        expected = self.expected_counts_by_domain(domains, test_ratio, val_ratio)
        for domain in domains.unique():
            assert (
                sum(1 for k in test_keys if domains.loc[k] == domain)
                == expected[domain]["test"]
            )
            assert (
                sum(1 for k in val_keys if domains.loc[k] == domain)
                == expected[domain]["val"]
            )
            assert (
                sum(1 for k in train_keys if domains.loc[k] == domain)
                == expected[domain]["train"]
            )

    def test_deterministic_with_seed(self):
        indices = [f"x{j}" for j in range(1, 21)]
        labels = ["X"] * 10 + ["Y"] * 5 + ["Z"] * 5
        domains = pd.Series(labels, index=indices)

        a = splitting.stratified_splitter(domains, 0.2, 0.25, random_seed=2025)
        b = splitting.stratified_splitter(domains, 0.2, 0.25, random_seed=2025)

        # exact same partitions for same seed
        assert a == b

    def test_keys_are_returned_sorted(self):
        indices = [f"p{j}" for j in range(1, 21)]
        labels = ["X"] * 10 + ["Y"] * 5 + ["Z"] * 5
        domains = pd.Series(labels, index=indices)

        train_keys, val_keys, test_keys = splitting.stratified_splitter(
            domains, 0.2, 0.3, random_seed=42
        )

        assert train_keys == sorted(train_keys)
        assert val_keys == sorted(val_keys)
        assert test_keys == sorted(test_keys)


class TestCrossValidationTestSplitter:
    def expected_split_sizes(self, n_items: int, n_splits: int) -> set[int]:
        base = n_items // n_splits
        remainder = n_items % n_splits
        sizes = {base}
        if remainder:
            sizes.add(base + 1)
        return sizes

    def test_stratified_counts_and_partitioning(self):
        indices = [f"i{j}" for j in range(1, 13)]
        labels = ["A", "A", "A", "A", "B", "B", "B", "C", "C", "D", "D", "D"]
        domains = pd.Series(labels, index=indices)

        n_splits = 3

        splits = splitting.cross_validation_test_splitter(
            domains, n_splits, random_seed=123
        )

        # types
        assert isinstance(splits, list)
        assert len(splits) == n_splits
        assert all(isinstance(split, list) for split in splits)
        assert all(isinstance(k, str) for split in splits for k in split)

        # disjointness and coverage
        all_keys = set().union(*[set(split) for split in splits])
        assert all_keys == set(indices)
        for i in range(n_splits):
            for j in range(i + 1, n_splits):
                assert set(splits[i]).isdisjoint(set(splits[j]))

        # per-domain counts are balanced (diff at most 1)
        for domain in domains.unique():
            domain_keys = domains[domains == domain].index.to_list()
            expected_sizes = self.expected_split_sizes(len(domain_keys), n_splits)
            sizes = [
                sum(1 for k in split if domains.loc[k] == domain) for split in splits
            ]
            assert all(size in expected_sizes for size in sizes)

    def test_deterministic_with_seed(self):
        indices = [f"x{j}" for j in range(1, 21)]
        labels = ["X"] * 10 + ["Y"] * 5 + ["Z"] * 5
        domains = pd.Series(labels, index=indices)

        a = splitting.cross_validation_test_splitter(domains, 4, random_seed=1337)
        b = splitting.cross_validation_test_splitter(domains, 4, random_seed=1337)

        assert a == b

    def test_keys_are_returned_sorted(self):
        indices = [f"p{j}" for j in range(1, 21)]
        labels = ["X"] * 10 + ["Y"] * 5 + ["Z"] * 5
        domains = pd.Series(labels, index=indices)

        splits = splitting.cross_validation_test_splitter(domains, 5, random_seed=42)

        assert all(split == sorted(split) for split in splits)


class TestTrainSplitAfterTest:
    def expected_counts_by_domain(
        self, domains: pd.Series, test_keys: list[str], val_ratio: float
    ) -> dict[str, dict[str, int]]:
        remaining = domains.drop(index=test_keys)
        expected: dict[str, dict[str, int]] = {}
        for domain in remaining.unique():
            n = int((remaining == domain).sum())
            n_all = int((domains == domain).sum())
            val_k = int(np.ceil(n_all * val_ratio))
            expected[domain] = {
                "n": n,
                "val": val_k,
                "train": n - val_k,
            }
        return expected

    def test_stratified_counts_and_partitioning(self):
        indices = [f"i{j}" for j in range(1, 14)]
        labels = ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"]
        domains = pd.Series(labels, index=indices)

        test_keys = ["i2", "i6", "i9"]
        val_ratio = 0.3

        train_keys, val_keys = splitting.train_split_after_test(
            domains, test_keys, val_ratio=val_ratio, random_seed=123
        )

        # types
        assert isinstance(train_keys, list)
        assert isinstance(val_keys, list)
        assert all(isinstance(k, str) for k in train_keys)
        assert all(isinstance(k, str) for k in val_keys)

        # disjointness and coverage (excluding test keys)
        remaining_keys = set(indices) - set(test_keys)
        assert set(train_keys).isdisjoint(set(val_keys))
        assert set(train_keys) | set(val_keys) == remaining_keys

        # per-domain counts match the ceil-based logic used by the function
        expected = self.expected_counts_by_domain(domains, test_keys, val_ratio)
        remaining = domains.drop(index=test_keys)
        for domain in remaining.unique():
            assert (
                domains[val_keys][domains[val_keys] == domain].size
                == expected[domain]["val"]
            )
            assert (
                domains[train_keys][domains[train_keys] == domain].size
                == expected[domain]["train"]
            )

    def test_deterministic_with_seed(self):
        indices = [f"x{j}" for j in range(1, 21)]
        labels = ["X"] * 10 + ["Y"] * 5 + ["Z"] * 5
        domains = pd.Series(labels, index=indices)

        test_keys = ["x1", "x12", "x17"]

        a = splitting.train_split_after_test(
            domains, test_keys, val_ratio=0.25, random_seed=2025
        )
        b = splitting.train_split_after_test(
            domains, test_keys, val_ratio=0.25, random_seed=2025
        )

        assert a == b

    def test_keys_are_returned_sorted(self):
        indices = [f"p{j}" for j in range(1, 21)]
        labels = ["X"] * 10 + ["Y"] * 5 + ["Z"] * 5
        domains = pd.Series(labels, index=indices)

        test_keys = ["p2", "p11", "p18"]

        train_keys, val_keys = splitting.train_split_after_test(
            domains, test_keys, val_ratio=0.2, random_seed=42
        )

        assert train_keys == sorted(train_keys)
        assert val_keys == sorted(val_keys)


class TestCrossValidationSplitter:
    def expected_counts_by_domain(
        self, domains: pd.Series, test_keys: list[str], val_ratio: float
    ) -> dict[str, dict[str, int]]:
        remaining = domains.drop(index=test_keys)
        expected: dict[str, dict[str, int]] = {}
        for domain in remaining.unique():
            n = int((remaining == domain).sum())
            n_all = int((domains == domain).sum())
            val_k = int(np.ceil(n_all * val_ratio))
            expected[domain] = {
                "n": n,
                "val": val_k,
                "train": n - val_k,
            }
        return expected

    def test_stratified_counts_and_partitioning(self):
        indices = [f"i{j}" for j in range(1, 14)]
        labels = ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"]
        domains = pd.Series(labels, index=indices)

        n_splits = 3
        val_ratio = 0.1

        splits = splitting.cross_validation_splitter(
            domains, n_splits, val_ratio=val_ratio, random_seed=123
        )

        # types
        assert isinstance(splits, list)
        assert len(splits) == n_splits
        assert all(isinstance(split, dict) for split in splits)
        assert all(set(split.keys()) == {"train", "val", "test"} for split in splits)
        assert all(isinstance(k, str) for split in splits for k in split["train"])
        assert all(isinstance(k, str) for split in splits for k in split["val"])
        assert all(isinstance(k, str) for split in splits for k in split["test"])

        # disjointness and coverage per fold
        for split in splits:
            train_keys = split["train"]
            val_keys = split["val"]
            test_keys = split["test"]
            all_keys = set(train_keys) | set(val_keys) | set(test_keys)
            assert all_keys == set(indices)
            assert set(train_keys).isdisjoint(set(val_keys))
            assert set(train_keys).isdisjoint(set(test_keys))
            assert set(val_keys).isdisjoint(set(test_keys))

            expected = self.expected_counts_by_domain(domains, test_keys, val_ratio)
            remaining = domains.drop(index=test_keys)
            for domain in remaining.unique():
                assert (
                    domains[val_keys][domains[val_keys] == domain].size
                    == expected[domain]["val"]
                )
                assert (
                    domains[train_keys][domains[train_keys] == domain].size
                    == expected[domain]["train"]
                )

    def test_deterministic_with_seed(self):
        indices = [f"x{j}" for j in range(1, 21)]
        labels = ["X"] * 10 + ["Y"] * 5 + ["Z"] * 5
        domains = pd.Series(labels, index=indices)

        a = splitting.cross_validation_splitter(
            domains, 4, val_ratio=0.25, random_seed=1337
        )
        b = splitting.cross_validation_splitter(
            domains, 4, val_ratio=0.25, random_seed=1337
        )

        assert a == b

    def test_keys_are_returned_sorted(self):
        indices = [f"p{j}" for j in range(1, 21)]
        labels = ["X"] * 10 + ["Y"] * 5 + ["Z"] * 5
        domains = pd.Series(labels, index=indices)

        splits = splitting.cross_validation_splitter(
            domains, 5, val_ratio=0.2, random_seed=42
        )

        assert all(split["train"] == sorted(split["train"]) for split in splits)
        assert all(split["val"] == sorted(split["val"]) for split in splits)
        assert all(split["test"] == sorted(split["test"]) for split in splits)
