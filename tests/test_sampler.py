import numpy as np

from dr_sad.data import sampler


class TestStratifiedSampling:
    def test_even_no_shuffle(self):
        domains = [0, 0, 1, 1, 2, 2]
        order = sampler.stratified_sampling(domains, shuffle=False)
        assert order.tolist() == [0, 2, 4, 1, 3, 5]

    def test_uneven_no_shuffle(self):
        domains = [0, 0, 1, 1, 1, 2, 2]
        order = sampler.stratified_sampling(domains, shuffle=False)
        assert order.tolist() == [2, 0, 5, 3, 1, 6, 4]

    def test_even_with_shuffle(self):
        domains = [0, 0, 1, 1, 2, 2]
        order = sampler.stratified_sampling(domains, shuffle=True)
        order_domain = [domains[i] for i in order]
        assert order_domain == [0, 1, 2, 0, 1, 2]

    def test_uneven_with_shuffle(self):
        domains = [0, 0, 1, 1, 1, 2, 2]
        order = sampler.stratified_sampling(domains, shuffle=True)
        order_domain = [domains[i] for i in order]
        assert order_domain == [1, 0, 2, 1, 0, 2, 1]

    def test_single_domain(self):
        domains = [0, 0, 0, 0]
        order = sampler.stratified_sampling(domains, shuffle=False)
        assert order.tolist() == [0, 1, 2, 3]

    def test_missing_domain(self):
        domains = [0, 0, 2, 2]
        order = sampler.stratified_sampling(domains, shuffle=True)
        order_domain = [domains[i] for i in order]
        assert order_domain == [0, 2, 0, 2]


class TestStratifiedSampler:
    def test_even_no_shuffle(self):
        domains = [0, 0, 1, 1, 2, 2]
        s = sampler.StratifiedSampler(domains, shuffle=False)
        order = list(iter(s))
        assert order == [0, 2, 4, 1, 3, 5]

    def test_uneven_no_shuffle(self):
        domains = [0, 0, 1, 1, 1, 2, 2]
        s = sampler.StratifiedSampler(domains, shuffle=False)
        order = list(iter(s))
        assert order == [2, 0, 5, 3, 1, 6, 4]

    def test_even_with_shuffle(self):
        domains = [0, 0, 1, 1, 2, 2]
        s = sampler.StratifiedSampler(domains, shuffle=True)
        order = list(iter(s))
        order_domain = [domains[i] for i in order]
        assert order_domain == [0, 1, 2, 0, 1, 2]

    def test_uneven_with_shuffle(self):
        domains = [0, 0, 1, 1, 1, 2, 2]
        s = sampler.StratifiedSampler(domains, shuffle=True)
        order = list(iter(s))
        order_domain = [domains[i] for i in order]
        assert order_domain == [1, 0, 2, 1, 0, 2, 1]

    def test_single_domain(self):
        domains = [0, 0, 0, 0]
        s = sampler.StratifiedSampler(domains, shuffle=False)
        order = list(iter(s))
        assert order == [0, 1, 2, 3]

    def test_missing_domain(self):
        domains = [0, 0, 2, 2]
        s = sampler.StratifiedSampler(domains, shuffle=True)
        order = list(iter(s))
        order_domain = [domains[i] for i in order]
        assert order_domain == [0, 2, 0, 2]

    def test_with_generator(self):
        np_rng = np.random.RandomState(42)
        domains = [0, 0, 1, 1, 2, 2]
        s = sampler.StratifiedSampler(domains, shuffle=True, generator=np_rng)
        order = list(iter(s))
        order_domain = [domains[i] for i in order]
        assert order_domain == [0, 1, 2, 0, 1, 2]

    def text_length(self):
        domains = [0, 0, 1, 1, 2, 2]
        s = sampler.StratifiedSampler(domains, shuffle=True)
        assert len(s) == len(domains)
