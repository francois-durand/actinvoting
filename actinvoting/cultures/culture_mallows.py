import numpy as np
from actinvoting.cultures.culture import Culture
from actinvoting.util import kendall_tau_id_ranking, kendall_tau_id_borda, borda_from_ranking


class CultureMallows(Culture):
    """
    Mallows Culture.

    The pole is ranking = [0, 1, 2, 3, ...], i.e. borda = [m-1, m-2, ...].

    Examples
    --------
    Usual case:

        >>> culture = CultureMallows(m=3, phi=.5, seed=42)
        >>> profile = culture.random_profile(n=10000)
        >>> print(profile)
        Profile((0, 1, 2): 3766,
                (0, 2, 1): 1930,
                (1, 0, 2): 1958,
                (1, 2, 0): 944,
                (2, 0, 1): 946,
                (2, 1, 0): 456)

    Note that `(0, 1, 2)` is the most frequent, `(0, 2, 1)` and `(1, 0, 2)` (each at distance 1) are half as frequent
    because `phi = .5`, `(1, 2, 0)` and `(2, 0, 1)` (each at distance 2) are 1/4 as frequent, and `(2, 1, 0)` (at
    distance 3) is 1/8 as frequent.

    Particular case of a Dirac:

        >>> culture = CultureMallows(m=6, phi=0)
        >>> list(culture.powers_of_phi)
        [1, 0, 0, 0, 0, 0]
        >>> list(culture.powers_of_phi_cumsum)
        [1, 1, 1, 1, 1, 1]
        >>> for candidate, insertion_probas in culture.d_candidate_insertion_probas.items():
        ...     print(candidate, list(insertion_probas))
        0 [1.0]
        1 [1.0, 0.0]
        2 [1.0, 0.0, 0.0]
        3 [1.0, 0.0, 0.0, 0.0]
        4 [1.0, 0.0, 0.0, 0.0, 0.0]
        5 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> culture.normalization_constant
        1
        >>> culture.proba_ranking([0, 1, 2, 3, 4, 5])
        1.0
        >>> culture.proba_ranking([2, 5, 0, 1, 3, 4])
        0.0
        >>> list(culture.random_ranking())
        [0, 1, 2, 3, 4, 5]

    Particular case of the Impartial Culture:

        >>> culture = CultureMallows(m=6, phi=1, seed=42)
        >>> list(culture.powers_of_phi)
        [1, 1, 1, 1, 1, 1]
        >>> list(culture.powers_of_phi_cumsum)
        [1, 2, 3, 4, 5, 6]
        >>> for candidate, insertion_probas in culture.d_candidate_insertion_probas.items():
        ...     print(candidate, list(insertion_probas))
        0 [1.0]
        1 [0.5, 0.5]
        2 [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
        3 [0.25, 0.25, 0.25, 0.25]
        4 [0.2, 0.2, 0.2, 0.2, 0.2]
        5 [0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666]
        >>> culture.normalization_constant
        720
        >>> culture.proba_ranking([0, 1, 2, 3, 4, 5])
        0.001388888888888889
        >>> culture.proba_ranking([2, 5, 0, 1, 3, 4])
        0.001388888888888889
        >>> list(culture.random_ranking())
        [5, 2, 3, 0, 1, 4]

    References
    ----------
    Doignon, J. P., Pekeč, A., & Regenwetter, M. (2004). The repeated insertion model for rankings: Missing link
    between two subset choice models. Psychometrika, 69(1), 33-54.

    Lu, T., & Boutilier, C. (2014). Effective sampling and learning for Mallows models with pairwise-preference data.
    J. Mach. Learn. Res., 15(1), 3783-3829.
    """

    def __init__(self, m, phi, seed=None):
        super().__init__(m=m, seed=seed)
        self.phi = phi
        self.powers_of_phi = phi**np.arange(m)
        self.powers_of_phi_cumsum = self.powers_of_phi.cumsum()
        self.d_candidate_insertion_probas = {
            candidate: self.powers_of_phi[:candidate + 1] / self.powers_of_phi_cumsum[candidate]
            for candidate in range(self.m)
        }
        self.normalization_constant = np.prod(self.powers_of_phi_cumsum)

    def proba_ranking(self, ranking):
        return self.phi ** kendall_tau_id_ranking(ranking) / self.normalization_constant

    def proba_borda(self, borda):
        return self.phi ** kendall_tau_id_borda(borda) / self.normalization_constant

    def random_ranking(self):
        ranking_worst_to_best = []
        for candidate in range(self.m):
            insertion_index = self.rng.choice(
                candidate + 1, size=1, p=self.d_candidate_insertion_probas[candidate]
            )[0]
            ranking_worst_to_best.insert(insertion_index, candidate)
        return np.array(ranking_worst_to_best[::-1])

    def random_borda(self):
        return borda_from_ranking(self.random_ranking())

    def random_profile(self, n):
        return self._random_profile_using_random_ranking(n)
