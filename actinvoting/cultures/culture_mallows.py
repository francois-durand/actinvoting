import numpy as np
import sympy

from actinvoting.cultures.culture import Culture
from actinvoting.util import kendall_tau_id_ranking, kendall_tau_id_borda, borda_from_ranking
from actinvoting.util_cache import cached_property


class CultureMallows(Culture):
    """
    Mallows Culture.

    The pole is ranking = [0, 1, 2, 3, ...], i.e. borda = [m-1, m-2, ...].

    Examples
    --------
    Usual case:

        >>> culture = CultureMallows(m=3, phi=sympy.Rational(1, 2), seed=42)
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

        >>> culture = CultureMallows(m=6, phi=sympy.Integer(0))
        >>> list(culture.powers_of_phi)
        [1, 0, 0, 0, 0, 0]
        >>> list(culture.powers_of_phi_cumsum)
        [1, 1, 1, 1, 1, 1]
        >>> for candidate, insertion_probas in culture.d_candidate_insertion_probas.items():
        ...     print(candidate, list(insertion_probas))
        0 [1]
        1 [1, 0]
        2 [1, 0, 0]
        3 [1, 0, 0, 0]
        4 [1, 0, 0, 0, 0]
        5 [1, 0, 0, 0, 0, 0]
        >>> culture.normalization_constant
        1
        >>> culture.proba_ranking([0, 1, 2, 3, 4, 5])
        1
        >>> culture.proba_ranking([2, 5, 0, 1, 3, 4])
        0
        >>> culture.proba_borda([5, 4, 3, 2, 1, 0])
        1
        >>> culture.proba_borda([3, 2, 5, 1, 0, 4])
        0
        >>> list(culture.random_ranking())
        [0, 1, 2, 3, 4, 5]
        >>> list(culture.random_borda())
        [5, 4, 3, 2, 1, 0]

    Particular case of the Impartial Culture:

        >>> culture = CultureMallows(m=6, phi=sympy.Integer(1), seed=42)
        >>> list(culture.powers_of_phi)
        [1, 1, 1, 1, 1, 1]
        >>> list(culture.powers_of_phi_cumsum)
        [1, 2, 3, 4, 5, 6]
        >>> for candidate, insertion_probas in culture.d_candidate_insertion_probas.items():
        ...     print(candidate, list(insertion_probas))
        0 [1]
        1 [1/2, 1/2]
        2 [1/3, 1/3, 1/3]
        3 [1/4, 1/4, 1/4, 1/4]
        4 [1/5, 1/5, 1/5, 1/5, 1/5]
        5 [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
        >>> culture.normalization_constant
        720
        >>> culture.proba_ranking([0, 1, 2, 3, 4, 5])
        1/720
        >>> culture.proba_ranking([2, 5, 0, 1, 3, 4])
        1/720
        >>> culture.proba_borda([5, 4, 3, 2, 1, 0])
        1/720
        >>> culture.proba_borda([3, 2, 5, 1, 0, 4])
        1/720
        >>> list(culture.random_ranking())
        [5, 2, 3, 0, 1, 4]
        >>> list(culture.random_borda())
        [3, 4, 0, 2, 1, 5]

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

    @cached_property
    def powers_of_phi(self):
        return self.phi**np.arange(self.m)

    @cached_property
    def powers_of_phi_cumsum(self):
        return self.powers_of_phi.cumsum()

    @cached_property
    def d_candidate_insertion_probas(self):
        return {
            candidate: self.powers_of_phi[:candidate + 1] / self.powers_of_phi_cumsum[candidate]
            for candidate in range(self.m)
        }

    @cached_property
    def d_candidate_insertion_probas_as_floats(self):
        return {
            candidate: np.array(insertion_probas, dtype=float)
            for candidate, insertion_probas in self.d_candidate_insertion_probas.items()
        }

    @cached_property
    def normalization_constant(self):
        return np.prod(self.powers_of_phi_cumsum)

    def proba_ranking(self, ranking):
        return self.phi ** kendall_tau_id_ranking(ranking) / self.normalization_constant

    def proba_borda(self, borda):
        return self.phi ** kendall_tau_id_borda(borda) / self.normalization_constant

    def random_ranking(self):
        # We use the Repeated Insertion Model or RIM (cf. references in the docstring of the class).
        ranking_worst_to_best = []
        for candidate in range(self.m):
            insertion_index = self.rng.choice(
                candidate + 1, size=1, p=self.d_candidate_insertion_probas_as_floats[candidate]
            )[0]
            ranking_worst_to_best.insert(insertion_index, candidate)
        return np.array(ranking_worst_to_best[::-1])

    def random_borda(self):
        return borda_from_ranking(self.random_ranking())

    def random_profile(self, n):
        return self._random_profile_using_random_ranking(n)

    @cached_property
    def average_profile(self):
        return self._average_profile_using_proba_ranking
