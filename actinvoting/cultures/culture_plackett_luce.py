import numpy as np
from actinvoting.cultures.culture import Culture
from actinvoting.util import ranking_from_borda, borda_from_ranking
from actinvoting.util_cache import cached_property


class CulturePlackettLuce(Culture):
    """
    Placket-Luce Culture.

    Examples
    --------
        >>> culture = CulturePlackettLuce(values=[.5, .7, .3, .2, .1, .2], seed=42)
        >>> culture.m
        6
        >>> list(culture.values_normalized)
        [0.25, 0.35, 0.15, 0.1, 0.05, 0.1]
        >>> culture.proba_ranking([2, 5, 0, 1, 3, 4])
        0.002745098039215686
        >>> culture.proba_borda([3, 2, 5, 1, 0, 4])
        0.002745098039215686
        >>> list(culture.random_ranking())
        [3, 1, 4, 2, 0, 5]
        >>> list(culture.random_borda())
        [4, 3, 1, 5, 0, 2]
    """

    def __init__(self, values, seed):
        super().__init__(m=len(values), seed=seed)
        self.values = np.array(values)

    @cached_property
    def values_normalized(self):
        return self.values / self.values.sum()

    def proba_ranking(self, ranking):
        return np.prod(self.values[ranking[::-1]] / np.cumsum(self.values[ranking[::-1]]))

    def proba_borda(self, borda):
        return self.proba_ranking(ranking_from_borda(borda))

    def random_ranking(self):
        return self.rng.choice(self.m, size=self.m, replace=False, p=self.values_normalized)

    def random_borda(self):
        return borda_from_ranking(self.random_ranking())

    def random_profile(self, n):
        return self._random_profile_using_random_ranking(n)
