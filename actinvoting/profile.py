from collections import defaultdict

import numpy as np

from actinvoting.util import ranking_from_borda, borda_from_ranking
from actinvoting.util_cache import cached_property


class Profile:
    """
    A voting profile.

    Examples
    --------
    A profile can be represented in several ways: a dictionary associating ranking to multiplicities (number of voters),
    an array of unique rankings and the corresponding vector of multiplicities, as well as both these possibilities
    with Borda vectors instead of rankings.

    You should avoid the default initialization method `Profile(...)`, because its syntax is quite likely to change
    over time. Instead, use the constructors which explicitly specify the input format:

        >>> Profile.from_d_ranking_multiplicity({(0, 1, 2): 3, (1, 0, 2): 2})
        Profile(d_ranking_multiplicity={(0, 1, 2): 3, (1, 0, 2): 2})

        >>> Profile.from_d_borda_multiplicity({(2, 1, 0): 3, (1, 2, 0): 2})
        Profile(d_ranking_multiplicity={(0, 1, 2): 3, (1, 0, 2): 2})

        >>> Profile.from_unique_rankings_and_multiplicities(
        ...     unique_rankings=[[0, 1, 2], [1, 0, 2]], multiplicities=[3, 2])
        Profile(d_ranking_multiplicity={(0, 1, 2): 3, (1, 0, 2): 2})

        >>> Profile.from_unique_bordas_and_multiplicities(
        ...     unique_bordas=[[2, 1, 0], [1, 2, 0]], multiplicities=[3, 2])
        Profile(d_ranking_multiplicity={(0, 1, 2): 3, (1, 0, 2): 2})

    In the case where there is an integer number of voters for each ranking, you can also list the rankings or the
    Borda vectors with repetitions:

        >>> Profile.from_rankings([
        ...     [0, 1, 2],
        ...     [0, 1, 2],
        ...     [0, 1, 2],
        ...     [1, 0, 2],
        ...     [1, 0, 2],
        ... ])
        Profile(d_ranking_multiplicity={(0, 1, 2): 3, (1, 0, 2): 2})

        >>> Profile.from_bordas([
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ...     [2, 1, 0],
        ...     [1, 2, 0],
        ...     [1, 2, 0],
        ... ])
        Profile(d_ranking_multiplicity={(0, 1, 2): 3, (1, 0, 2): 2})

    When you have a profile, you can access all these basic ways of representing it:

        >>> profile = Profile.from_d_ranking_multiplicity({(0, 1, 2): 3, (1, 0, 2): 2})
        >>> profile.d_borda_multiplicity
        {(2, 1, 0): 3, (1, 2, 0): 2}
        >>> profile.unique_rankings
        array([[0, 1, 2],
               [1, 0, 2]])
        >>> profile.unique_bordas  # DOCTEST: +ELLIPSIS
        array([[2, 1, 0],
               [1, 2, 0]])
        >>> profile.multiplicities
        array([3, 2])

    Features:

        >>> profile
        Profile(d_ranking_multiplicity={(0, 1, 2): 3, (1, 0, 2): 2})
        >>> print(profile)
        Profile((0, 1, 2): 3,
                (1, 0, 2): 2)
        >>> profile.n
        5
        >>> profile.m
        3
        >>> profile.weighted_majority_matrix
        array([[0, 3, 5],
               [2, 0, 5],
               [0, 0, 0]])
        >>> profile.majority_matrix
        array([[0., 1., 1.],
               [0., 0., 1.],
               [0., 0., 0.]])
        >>> profile.is_condorcet_winner
        array([ True, False, False])
        >>> profile.condorcet_winners
        [0]
        >>> profile.condorcet_winner
        0
        >>> profile.nb_condorcet_winners
        1
        >>> profile.exists_condorcet_winner
        True
        >>> profile.is_weak_condorcet_winner
        array([ True, False, False])
        >>> profile.weak_condorcet_winners
        [0]
        >>> profile.nb_weak_condorcet_winners
        1
        >>> profile.exists_condorcet_order
        True

    Majority matrix with a tie:

        >>> profile = Profile.from_d_ranking_multiplicity({(0, 1, 2): 2, (1, 0, 2): 2})
        >>> profile.weighted_majority_matrix
        array([[0, 2, 4],
               [2, 0, 4],
               [0, 0, 0]])
        >>> profile.majority_matrix
        array([[0. , 0.5, 1. ],
               [0.5, 0. , 1. ],
               [0. , 0. , 0. ]])
        >>> profile.is_condorcet_winner
        array([False, False, False])
        >>> profile.condorcet_winners
        []
        >>> profile.condorcet_winner
        -1
        >>> profile.nb_condorcet_winners
        0
        >>> profile.exists_condorcet_winner
        False
        >>> profile.is_weak_condorcet_winner
        array([ True,  True, False])
        >>> profile.weak_condorcet_winners
        [0, 1]
        >>> profile.nb_weak_condorcet_winners
        2
        >>> profile.exists_condorcet_order
        False
    """

    def __init__(self, d_ranking_multiplicity=None, d_borda_multiplicity=None):
        if d_ranking_multiplicity is None and d_borda_multiplicity is None:
            raise ValueError("Profile: you must specify `d_ranking_multiplicity` or `d_borda_multiplicity`.")
        self._d_borda_multiplicity = d_borda_multiplicity
        self._d_ranking_multiplicity = d_ranking_multiplicity

    # Constructors
    # ============

    @classmethod
    def from_d_ranking_multiplicity(cls, d_ranking_multiplicity):
        """
        New profile.

        Parameters
        ----------
        d_ranking_multiplicity: dict
            Key: ranking as a tuple. Value: multiplicity (number of voters).
        """
        return cls(d_ranking_multiplicity=dict(d_ranking_multiplicity))

    @classmethod
    def from_d_borda_multiplicity(cls, d_borda_multiplicity):
        """
        New profile.

        Parameters
        ----------
        d_borda_multiplicity: dict
            Key: ranking in Borda format as a tuple. Value: multiplicity (number of voters).
        """
        return cls(d_borda_multiplicity=dict(d_borda_multiplicity))

    @classmethod
    def from_unique_rankings_and_multiplicities(cls, unique_rankings, multiplicities):
        """
        New profile.

        Parameters
        ----------
        unique_rankings: List of List
            List of unique rankings.
        multiplicities: List
            Multiplicity (number of voters) corresponding to each unique ranking.
        """
        # On purpose, we do not record `unique_rankings` and `multiplicities` themselves.
        # This way, unique rankings will be sorted afterward (if accessed).
        return cls(d_ranking_multiplicity={
            tuple(ranking): multiplicity
            for ranking, multiplicity in zip(unique_rankings, multiplicities)
        })

    @classmethod
    def from_unique_bordas_and_multiplicities(cls, unique_bordas, multiplicities):
        """
        New profile.

        Parameters
        ----------
        unique_bordas: List of List
            List of unique rankings in Borda format.
        multiplicities: List
            Multiplicity (number of voters) corresponding to each unique ranking.
        """
        # Same here: we do not record `unique_bordas` and `multiplicities` themselves.
        return cls(d_borda_multiplicity={
            tuple(borda): multiplicity
            for borda, multiplicity in zip(unique_bordas, multiplicities)
        })

    @classmethod
    def from_rankings(cls, rankings):
        """
        New profile.

        Parameters
        ----------
        rankings: List of List
            List of rankings with possible repetitions.
        """
        d_ranking_multiplicity = defaultdict(int)
        for ranking in rankings:
            d_ranking_multiplicity[tuple(ranking)] += 1
        return cls(d_ranking_multiplicity=dict(d_ranking_multiplicity))

    @classmethod
    def from_bordas(cls, bordas):
        """
        New profile.

        Parameters
        ----------
        bordas: List of List
            List of rankings in Borda format, with possible repetitions.
        """
        d_borda_multiplicity = defaultdict(int)
        for borda in bordas:
            d_borda_multiplicity[tuple(borda)] += 1
        return cls(d_borda_multiplicity=dict(d_borda_multiplicity))

    # Basic properties
    # ================

    @property
    def d_borda_multiplicity(self):
        """
        dict: Key: ranking in Borda format as a tuple. Value: multiplicity (number of voters).
        """
        if self._d_borda_multiplicity is None:
            self._d_borda_multiplicity = {
                tuple(borda_from_ranking(ranking)): multiplicity
                for ranking, multiplicity in self.d_ranking_multiplicity.items()
            }
        return self._d_borda_multiplicity

    @property
    def d_ranking_multiplicity(self):
        """
        dict: Key: ranking as a tuple. Value: multiplicity (number of voters).
        """
        if self._d_ranking_multiplicity is None:
            self._d_ranking_multiplicity = {
                tuple(ranking_from_borda(borda)): multiplicity
                for borda, multiplicity in self.d_borda_multiplicity.items()
            }
        return self._d_ranking_multiplicity

    @cached_property
    def unique_rankings_and_multiplicities(self):
        unique_rankings = np.array(sorted(self.d_ranking_multiplicity.keys()))
        multiplicities = np.array([
            self.d_ranking_multiplicity[tuple(ranking)]
            for ranking in unique_rankings]
        )
        return unique_rankings, multiplicities

    @cached_property
    def unique_rankings(self):
        """
        ndarray: List of unique rankings.
        """
        return self.unique_rankings_and_multiplicities[0]

    @cached_property
    def multiplicities(self):
        """
        ndarray: Multiplicity (number of voters) corresponding to each ranking in `unique_rankings`.
        """
        return self.unique_rankings_and_multiplicities[1]

    @cached_property
    def unique_bordas(self):
        """
        ndarray: List of unique rankings in Borda format, in the same order as `unique_rankings`.
        """
        return np.array([borda_from_ranking(ranking) for ranking in self.unique_rankings])

    # Conversion to string
    # ====================

    def __repr__(self):
        d_rank_mult_sorted = {
            tuple(ranking): multiplicity
            for ranking, multiplicity in zip(self.unique_rankings, self.multiplicities)
        }
        return f"{type(self).__name__}(d_ranking_multiplicity={d_rank_mult_sorted})"

    def __str__(self):
        s = ',\n        '.join([
            f"{tuple(ranking)}: {multiplicity}"
            for ranking, multiplicity in zip(self.unique_rankings, self.multiplicities)
        ])
        return f"Profile(" + s + ")"

    # Other properties

    @cached_property
    def n(self):
        """
        int: Number of voters.
        """
        return sum(self.d_borda_multiplicity.values())

    @cached_property
    def m(self):
        """
        int: Number of candidates.
        """
        first_borda_in_dict = next(iter(self.d_borda_multiplicity))
        return len(first_borda_in_dict)

    @cached_property
    def weighted_majority_matrix(self):
        """
        ndarray: Weighted majority matrix. Coefficient `(c, d)` is the number of voters who prefer candidate `c` to
        candidate `d`. By convention, diagonal coefficients are set to 0.
        """
        # wmm = np.zeros((self.m, self.m), int)
        # for borda, multiplicity in self.d_borda_multiplicity.items():
        #     borda = np.array(borda)
        #     wmm += (borda[:, np.newaxis] > borda[np.newaxis, :]) * multiplicity
        # return wmm
        return np.tensordot(
            self.multiplicities,
            self.unique_bordas[:, :, np.newaxis] > self.unique_bordas[:, np.newaxis, :],
            axes=1
        )

    @cached_property
    def majority_matrix(self):
        """
        ndarray: Majority matrix. Coefficient `(c, d)` is 1.0 if more voters prefer candidate `c` to `d` than the
        opposite, 0.5 in case of tie, and 0.0 in case of defeat. By convention, diagonal coefficients are set to 0.
        """
        mm = (
            (self.weighted_majority_matrix > self.weighted_majority_matrix.T)
            + .5 * (self.weighted_majority_matrix == self.weighted_majority_matrix.T)
        )
        mm[np.diag_indices(self.m)] = 0.
        return mm

    @cached_property
    def is_condorcet_winner(self):
        """
        ndarray: For each candidate `c`, the corresponding coefficient is True if `c` is the Condorcet winner.
        """
        # All opponents have defeats = the whole _column_ is 0.
        return np.all(self.majority_matrix == 0, 0)

    @cached_property
    def condorcet_winners(self):
        """
        List: Condorcet winners. This is a list of size 0 or 1.
        """
        return list(np.where(self.is_condorcet_winner)[0])

    @cached_property
    def condorcet_winner(self):
        """
        int: Condorcet winner. If there is no Condorcet winner, then -1 by convention.
        """
        if not self.condorcet_winners:
            return -1
        return self.condorcet_winners[0]

    @cached_property
    def nb_condorcet_winners(self):
        """
        int: Number of Condorcet winners. May be 0 or 1.
        """
        return len(self.condorcet_winners)

    @cached_property
    def exists_condorcet_winner(self):
        """
        bool: True if there exists a Condorcet winner.
        """
        return np.any(self.is_condorcet_winner)

    @cached_property
    def is_weak_condorcet_winner(self):
        """
        ndarray: For each candidate `c`, the corresponding coefficient is True if `c` is a weak Condorcet winner.
        """
        # All opponents have defeats or ties = the whole _column_ is 0 or 0.5.
        return np.all(self.majority_matrix <= .5, 0)

    @cached_property
    def weak_condorcet_winners(self):
        """
        List: Weak Condorcet winners.
        """
        return list(np.where(self.is_weak_condorcet_winner)[0])

    @cached_property
    def nb_weak_condorcet_winners(self):
        """
        int: Number of weak Condorcet winners.
        """
        return len(self.weak_condorcet_winners)

    @cached_property
    def exists_condorcet_order(self):
        """
        bool: True if the majority relation is transitive.
        """
        return len(set(self.majority_matrix.sum(axis=1))) == self.m
