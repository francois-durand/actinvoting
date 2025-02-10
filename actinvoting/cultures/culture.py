from collections import defaultdict
from itertools import permutations

import numpy as np

from actinvoting.profile import Profile
from actinvoting.util_cache import cached_property


class Culture:
    """
    Culture: probabilistic model to generate rankings.

    It is assumed here that it is a "General Independent Culture" (GIC), i.e, voters have independent preferences.

    Parameters
    ----------
    m: int
        Number of candidates.
    seed: int
        Random seed.
    """

    def __init__(self, m, seed=None):
        self.m = m
        self.rng = np.random.default_rng(seed)

    def proba_ranking(self, ranking):
        """
        Probability of a ranking.

        Parameters
        ----------
        ranking: List
            A ranking. E.g. [0, 2, 1, 3] corresponds to the preference ranking 0 > 2 > 1 > 3.

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
            A ranking in Borda format. E.g. [3, 1, 2, 0] corresponds to the preference ranking 0 > 2 > 1 > 3.

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
            A random ranking. E.g. [0, 2, 1, 3] corresponds to the preference ranking 0 > 2 > 1 > 3.
        """
        raise NotImplementedError

    def random_borda(self):
        """
        Random ranking in Borda format.

        Returns
        -------
        ndarray
            A random ranking in Borda format. E.g. [3, 1, 2, 0] corresponds to the preference ranking 0 > 2 > 1 > 3.
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
        # In a given subclass, this method typically calls either `_random_profile_using_random_ranking` or
        # `_random_profile_using_random_borda`.
        raise NotImplementedError

    @cached_property
    def _average_profile_using_proba_ranking(self):
        """
        Average profile, using `proba_ranking` as subroutine.

        Returns
        -------
        Profile
            A profile where the weight for each ranking is the corresponding probability in the culture.
        """
        return Profile.from_d_ranking_multiplicity({
            ranking: self.proba_ranking(np.array(ranking))
            for ranking in permutations(range(self.m))
        })

    @cached_property
    def _average_profile_using_proba_borda(self):
        """
        Average profile, using `proba_borda` as subroutine.

        Returns
        -------
        Profile
            A profile where the weight for each ranking is the corresponding probability in the culture.
        """
        return Profile.from_d_borda_multiplicity({
            borda: self.proba_borda(np.array(borda))
            for borda in permutations(range(self.m))
        })

    @cached_property
    def average_profile(self):
        """
        Average profile.

        Returns
        -------
        Profile
            A profile where the weight for each ranking is the corresponding probability in the culture.
        """
        # In a given subclass, this method typically calls either `_average_profile_using_proba_ranking` or
        # `_average_profile_using_proba_borda`.
        raise NotImplementedError

    def proba_high_low(self, c, higher, lower):
        """
        Probability that a random ranking places candidate `c` below certain adversaries and above the other ones.

        Parameters
        ----------
        c: int
            A candidate.
        higher: Set
            The adversaries that should be higher than `c`.
        lower: Set
            The adversaries that should be lower than `c`. Note that together, `c`, `higher` and `lower` must cover
            all the candidates.

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
