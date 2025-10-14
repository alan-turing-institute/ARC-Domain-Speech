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
