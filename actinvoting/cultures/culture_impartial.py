from math import factorial
from actinvoting.cultures.culture import Culture


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
    """

    def proba_ranking(self, ranking):
        return 1 / factorial(self.m)

    def proba_borda(self, borda):
        return 1 / factorial(self.m)

    def random_ranking(self):
        return self.rng.permutation(self.m)

    def random_borda(self):
        return self.rng.permutation(self.m)

    def random_profile(self, n):
        return self._random_profile_using_random_borda(n)
