import numpy as np
from torch.utils.data import Sampler


def stratified_sampling(
    domains: np.ndarray | list[int],
    shuffle: bool = True,
    generator: np.random.RandomState | None = None,
) -> np.ndarray:
    """This function returns indices that sample from the input domains in a stratified
    manner. If shuffle is True, the order within each domain is randomized.

    Args:
        domains (np.ndarray | list[int]): Array or list of domain identifiers for each
            item.
        shuffle (bool, optional): Whether to shuffle the order within each domain.
            Defaults to True.
        generator (np.random.Generator or np.random.RandomState, optional): Random
            number generator for shuffling. Defaults to np.random.RandomState.
    """
    if generator is None:
        generator = np.random.RandomState()
    domains = np.asarray(domains)
    indices = np.arange(len(domains))

    domain_ids = np.unique(domains)

    # buckets per domain
    groups = [indices[domains == dn] for dn in domain_ids]
    if shuffle:
        groups = [generator.permutation(g) for g in groups]

    # ranks in (0,1) for each item within its domain (exclusive of 0 and 1)
    ranks_list = [
        np.linspace(1.0 / len(g), 1.0, len(g), endpoint=False) for g in groups
    ]
    class_list = [np.full(len(g), dn, dtype=int) for dn, g in enumerate(groups)]

    all_idx = np.concatenate(groups)
    all_ranks = np.concatenate(ranks_list)
    all_cls = np.concatenate(class_list)

    # lexsort: primary key = rank in (0,1), secondary key = domain id
    order = np.lexsort((all_cls, all_ranks))
    return all_idx[order]


class StratifiedSampler(Sampler):  # type: ignore[misc]
    def __init__(
        self,
        domains: np.ndarray | list[int],
        shuffle: bool = True,
        generator: np.random.RandomState | None = None,
    ):
        """Sampler that samples elements in a stratified manner based on the provided
        domain identifiers. This is designed to be used with PyTorch DataLoader.

        Args:
            domains (np.ndarray | list[int]): Array or list of domain identifiers for
                each item.
            shuffle (bool, optional): Whether to shuffle the order within each domain.
                Defaults to True.
            generator (np.random.Generator or np.random.RandomState, optional): Random
                number generator for shuffling. Defaults to np.random.RandomState.
        """
        self.domains = np.asarray(domains)
        self.shuffle = shuffle
        self.generator = generator if generator is not None else np.random.RandomState()

    def __iter__(self):
        order = stratified_sampling(
            self.domains, shuffle=self.shuffle, generator=self.generator
        )
        return iter(order.tolist())

    def __len__(self):
        return len(self.domains)
