import numpy as np
import sympy

from actinvoting.cultures.culture import Culture
from actinvoting.util import ranking_from_borda, borda_from_ranking
from actinvoting.util_cache import cached_property


class CulturePlackettLuce(Culture):
    """
    Placket-Luce Culture.

    Examples
    --------
        >>> values = [
        ...     sympy.Rational(1, 2), sympy.Rational(7, 10), sympy.Rational(3, 10),
        ...     sympy.Rational(1, 5), sympy.Rational(1, 10), sympy.Rational(1, 5)
        ... ]
        >>> culture = CulturePlackettLuce(values=values, seed=42)
        >>> culture.m
        6
        >>> list(culture.values_normalized)
        [1/4, 7/20, 3/20, 1/10, 1/20, 1/10]
        >>> culture.proba_ranking([2, 5, 0, 1, 3, 4])
        7/2550
        >>> culture.proba_borda([3, 2, 5, 1, 0, 4])
        7/2550
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

    @cached_property
    def values_normalized_as_floats(self):
        return np.array(self.values_normalized, dtype=float)

    def proba_ranking(self, ranking):
        return np.prod(self.values[ranking[::-1]] / np.cumsum(self.values[ranking[::-1]]))

    def proba_borda(self, borda):
        return self.proba_ranking(ranking_from_borda(borda))

    def random_ranking(self):
        return self.rng.choice(self.m, size=self.m, replace=False, p=self.values_normalized_as_floats)

    def random_borda(self):
        return borda_from_ranking(self.random_ranking())

    def random_profile(self, n):
        return self._random_profile_using_random_ranking(n)
