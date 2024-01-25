from collections import defaultdict
import numpy as np
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
        float
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
        float
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
