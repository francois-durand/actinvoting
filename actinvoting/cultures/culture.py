from collections import defaultdict
from itertools import permutations

import numpy as np
import sympy
from more_itertools import powerset
from scipy.optimize import minimize

from actinvoting.profile import Profile


class Culture:
    """
    Culture: probabilistic model to generate rankings.

    It is assumed here that all voters are independent.

    Parameters
    ----------
    m: int
        Number of candidates.
    seed: int
        Random seed.
    """

    def __init__(self, m, seed=None):
        """

        """
        self.m = m
        self.rng = np.random.default_rng(seed)

    def proba_ranking(self, ranking):
        """
        Probability of a ranking.

        Parameters
        ----------
        ranking: List
            A ranking.

        Returns
        -------
        float or sympy expr
            The probability to draw this ranking.
        """
        raise NotImplementedError

    def proba_borda(self, borda):
        """
        Probability of a ranking, given in Borda format.

        Parameters
        ----------
        borda: List
            A ranking in Borda format.

        Returns
        -------
        float or sympy expr
            The probability to draw this ranking.
        """
        raise NotImplementedError

    def random_ranking(self):
        """
        Random ranking.

        Returns
        -------
        ndarray
            A random ranking.
        """
        raise NotImplementedError

    def random_borda(self):
        """
        Random ranking in Borda format.

        Returns
        -------
        ndarray
            A random ranking in Borda format.
        """
        raise NotImplementedError

    def _random_profile_using_random_borda(self, n):
        """
        Random profile, using `random_borda` as subroutine.

        Parameters
        ----------
        n: int
            Number of voters.

        Returns
        -------
        Profile
            A random profile.
        """
        d_borda_multiplicity = defaultdict(int)
        for _ in range(n):
            d_borda_multiplicity[tuple(self.random_borda())] += 1
        return Profile.from_d_borda_multiplicity(d_borda_multiplicity)

    def _random_profile_using_random_ranking(self, n):
        """
        Random profile, using `random_ranking` as subroutine.

        Parameters
        ----------
        n: int
            Number of voters.

        Returns
        -------
        Profile
            A random profile.
        """
        d_ranking_multiplicity = defaultdict(int)
        for _ in range(n):
            d_ranking_multiplicity[tuple(self.random_ranking())] += 1
        return Profile.from_d_ranking_multiplicity(d_ranking_multiplicity)

    def random_profile(self, n):
        """
        Random profile.

        Parameters
        ----------
        n: int
            Number of voters.

        Returns
        -------
        Profile
            A random profile.
        """
        # The default implementation uses `random_borda` as subroutine.
        return self._random_profile_using_random_borda(n)

    def proba_high_low(self, c, higher, lower):
        """
        Probability that a random ranking places candidate `c` below certain ones and above the other ones.

        Parameters
        ----------
        c: int
            A candidate.
        higher: Set
            The candidates that should be higher.
        lower: Set
            The candidates that should be lower. Note that together, `c`, `higher` and `lower` must cover all the
            candidates.

        Returns
        -------
        float
            Probability that for a random ranking in this culture, candidates from `higher` are higher than `c` and
            those from `lower` are lower than `c`.
        """
        return np.sum([
            self.proba_ranking([*ranking_higher, c, *ranking_lower])
            for ranking_higher in permutations(higher)
            for ranking_lower in permutations(lower)
        ])

    def polynom_of_duels(self, c):
        x = sympy.symarray("x", self.m)
        candidates = set(range(self.m))
        other_candidates = candidates - {c}
        return sympy.Add(*[
            self.proba_high_low(c, set(higher), other_candidates - set(higher)) * sympy.Mul(*[x[d] for d in higher])
            for higher in powerset(other_candidates)
        ])

    def rational_fraction_condorcet(self, c, alpha=None):
        if alpha is None:
            alpha = [sympy.Rational(1, 2)] * self.m
        x = sympy.symarray("x", self.m)
        candidates = set(range(self.m))
        other_candidates = candidates - {c}
        return self.polynom_of_duels(c) / sympy.Mul(*[x[d]**alpha[d] for d in other_candidates])

    def zeta(self, c, alpha=None):
        x = sympy.symarray("x", self.m)
        f = self.rational_fraction_condorcet(c=c, alpha=alpha)
        f_lambdified = sympy.lambdify([*x[:c], *x[c + 1:]], f, "numpy")

        def f_vector_input(v):
            return f_lambdified(*v)
        res = minimize(f_vector_input, [1.0] * (self.m - 1))
        return res.x

    def jacobian_of_p(self, c):
        x = sympy.symarray("x", self.m)
        p = self.polynom_of_duels(c=c)
        return sympy.tensor.array.derive_by_array(p, [*x[:c], *x[c + 1:]])

    def hessian_of_p(self, c):
        x = sympy.symarray("x", self.m)
        p = self.polynom_of_duels(c=c)
        return sympy.hessian(p, [*x[:c], *x[c + 1:]])
