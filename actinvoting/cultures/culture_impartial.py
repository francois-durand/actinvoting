from math import factorial
from actinvoting.cultures.culture import Culture
from actinvoting.util_cache import cached_property


class CultureImpartial(Culture):
    """
    Impartial Culture.

    Examples
    --------
        >>> culture = CultureImpartial(m=6, seed=42)
        >>> culture.proba_ranking([2, 5, 0, 1, 3, 4])
        0.001388888888888889
        >>> culture.proba_borda([3, 2, 5, 1, 0, 4])
        0.001388888888888889
        >>> list(culture.random_ranking())
        [3, 2, 5, 4, 1, 0]
        >>> list(culture.random_borda())
        [2, 4, 0, 1, 3, 5]
        >>> culture.proba_high_low(c=0, higher=set(), lower={1, 2, 3, 4, 5})
        0.16666666666666666
    """

    @cached_property
    def _proba_any_ranking(self):
        return 1 / factorial(self.m)

    def proba_ranking(self, ranking):
        return self._proba_any_ranking

    def proba_borda(self, borda):
        return self._proba_any_ranking

    def random_ranking(self):
        return self.rng.permutation(self.m)

    def random_borda(self):
        return self.rng.permutation(self.m)

    def random_profile(self, n):
        return self._random_profile_using_random_borda(n)

    def proba_high_low(self, c, higher, lower):
        return factorial(len(higher)) * factorial(len(lower)) / factorial(self.m)
