import sympy

from actinvoting.cultures.culture import Culture
from actinvoting.util_cache import cached_property


class CultureImpartial(Culture):
    """
    Impartial Culture.

    Parameters
    ----------
    m: int
        Number of candidates.
    seed: int
        Random seed.

    Examples
    --------
        >>> culture = CultureImpartial(m=6, seed=42)
        >>> culture.proba_ranking([2, 5, 0, 1, 3, 4])
        1/720
        >>> culture.proba_borda([3, 2, 5, 1, 0, 4])
        1/720
        >>> culture.random_ranking()
        array([3, 2, 5, 4, 1, 0])
        >>> culture.random_borda()
        array([2, 4, 0, 1, 3, 5])
        >>> print(culture.random_profile(n=3))
        Profile((0, 1, 3, 2, 5, 4): 1,
                (0, 2, 4, 3, 5, 1): 1,
                (4, 0, 3, 2, 5, 1): 1)
        >>> culture.average_profile.exists_condorcet_order
        False
        >>> culture.proba_high_low(c=0, higher=set(), lower={1, 2, 3, 4, 5})
        1/6
    """

    def __repr__(self):
        return f"IC_{self.m=}"

    @cached_property
    def _proba_any_ranking(self):
        return 1 / sympy.factorial(self.m)

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

    @cached_property
    def average_profile(self):
        return self._average_profile_using_proba_ranking

    def proba_high_low(self, c, higher, lower):
        return sympy.factorial(len(higher)) * sympy.factorial(len(lower)) / sympy.factorial(self.m)
