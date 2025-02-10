import numpy as np
import sympy

from actinvoting.cultures.culture import Culture
from actinvoting.util_cache import cached_property


class CulturePerturbed(Culture):
    """
    Perturbed Culture.

    The pole (a.k.a. reference ranking) is ranking = [0, 1, 2, 3, ...], i.e. borda = [m-1, m-2, ...].

    Parameters
    ----------
    m: int
        Number of candidates.
    theta: sympy.Rational
        Concentration parameter, giving the additional probability of the pole. For theta = 1, the culture is a Dirac.
        For theta = 0, it is the Impartial Culture.
    seed: int
        Random seed.

    Examples
    --------
    Usual case:

        >>> culture = CulturePerturbed(m=3, theta=sympy.Rational(1, 2), seed=42)
        >>> profile = culture.random_profile(n=10000)
        >>> print(profile)
        Profile((0, 1, 2): 5778,
                (0, 2, 1): 864,
                (1, 0, 2): 883,
                (1, 2, 0): 830,
                (2, 0, 1): 835,
                (2, 1, 0): 810)

    As expected, there are approximately 5000 voters `(0, 1, 2)` due to the Dirac part, and the other 5000 voters
    are approximately equally shared between all rankings, including the pole.

        >>> culture.proba_high_low(c=0, higher={}, lower={1, 2})
        2/3
        >>> culture.proba_high_low(c=1, higher={}, lower={0, 2})
        1/6

    Particular case of a Dirac:

        >>> culture = CulturePerturbed(m=6, theta=1)
        >>> culture.proba_ranking([0, 1, 2, 3, 4, 5])
        1
        >>> culture.proba_ranking([2, 5, 0, 1, 3, 4])
        0
        >>> culture.proba_borda([5, 4, 3, 2, 1, 0])
        1
        >>> culture.proba_borda([3, 2, 5, 1, 0, 4])
        0
        >>> culture.random_ranking()
        array([0, 1, 2, 3, 4, 5])
        >>> culture.random_borda()
        array([5, 4, 3, 2, 1, 0])

    Particular case of the Impartial Culture:

        >>> culture = CulturePerturbed(m=6, theta=0, seed=42)
        >>> culture.proba_ranking([0, 1, 2, 3, 4, 5])
        1/720
        >>> culture.proba_ranking([2, 5, 0, 1, 3, 4])
        1/720
        >>> culture.proba_borda([5, 4, 3, 2, 1, 0])
        1/720
        >>> culture.proba_borda([3, 2, 5, 1, 0, 4])
        1/720
        >>> culture.random_ranking()
        array([3, 4, 2, 0, 5, 1])
        >>> culture.random_borda()
        array([2, 5, 0, 3, 1, 4])
    """

    def __init__(self, m, theta, seed=None):
        super().__init__(m=m, seed=seed)
        self.theta = theta

    def __repr__(self):
        return f"Perturbed_{self.m=}_{self.theta=}"

    @cached_property
    def _proba_pole(self):
        """
        Probability of the pole.

        Returns
        -------
        sympy.Rational
            Probability of the pole.
        """
        return self.theta + (1 - self.theta) / sympy.factorial(self.m)

    @cached_property
    def _proba_other_ranking(self):
        """
        Probability of a ranking that is not the pole.

        Returns
        -------
        sympy.Rational
            Probability of a ranking that is not the pole.
        """
        return (1 - self.theta) / sympy.factorial(self.m)

    @cached_property
    def _pole_ranking(self):
        """
        Pole given as a ranking.

        Returns
        -------
        ndarray
            Pole given as a ranking: [0, 1, 2, 3, ...].
        """
        return np.arange(self.m)

    @cached_property
    def _pole_borda(self):
        """
        Pole given as a borda vector.

        Returns
        -------
        ndarray
            Pole given as a borda vector: [m-1, m-2, ...].
        """
        return np.arange(self.m - 1, -1, -1)

    def proba_ranking(self, ranking):
        if np.array_equal(ranking, self._pole_ranking):
            return self._proba_pole
        else:
            return self._proba_other_ranking

    def proba_borda(self, borda):
        if np.array_equal(borda, self._pole_borda):
            return self._proba_pole
        else:
            return self._proba_other_ranking

    def random_ranking(self):
        if self.rng.random() < self.theta:
            return self._pole_ranking
        else:
            return self.rng.permutation(self.m)

    def random_borda(self):
        if self.rng.random() < self.theta:
            return self._pole_borda
        else:
            return self.rng.permutation(self.m)

    def random_profile(self, n):
        return self._random_profile_using_random_borda(n)

    @cached_property
    def average_profile(self):
        return self._average_profile_using_proba_ranking

    def proba_high_low(self, c, higher, lower):
        proba = sympy.factorial(len(higher)) * sympy.factorial(len(lower)) * self._proba_other_ranking
        if len(higher) == c and all([h < c for h in higher]):
            proba += self.theta
        return proba
