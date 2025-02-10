from actinvoting.cultures.culture import Culture
from actinvoting.profile import Profile
from actinvoting.util_cache import cached_property


class CultureFromProfile(Culture):
    """
    Culture deduced from an average profile.

    Parameters
    ----------
    base_profile: Profile
        Average profile. To each ranking, it associates the corresponding probability.
    seed: int
        Random seed.

    Examples
    --------
    >>> profile = Profile.from_d_ranking_multiplicity({(0, 1, 2): .5, (1, 0, 2): .5})
    >>> culture = CultureFromProfile(profile, seed=42)
    >>> culture.proba_ranking((0, 1, 2))
    0.5
    >>> culture.proba_borda((2, 1, 0))
    0.5
    >>> culture.random_ranking()
    array([1, 0, 2])
    >>> culture.random_borda()
    array([2, 1, 0])
    >>> print(culture.random_profile(n=3))
    Profile((0, 1, 2): 1,
            (1, 0, 2): 2)
    >>> print(culture.average_profile)
    Profile((0, 1, 2): 0.5,
            (1, 0, 2): 0.5)
    >>> culture.proba_high_low(c=0, higher={1}, lower={2})
    np.float64(0.5)
    """

    def __init__(self, base_profile, seed=None):
        super().__init__(m=base_profile.m, seed=seed)
        self.base_profile = base_profile

    def proba_ranking(self, ranking):
        try:
            return self.average_profile.d_ranking_multiplicity[tuple(ranking)]
        except KeyError:
            return 0

    def proba_borda(self, borda):
        try:
            return self.average_profile.d_borda_multiplicity[tuple(borda)]
        except KeyError:
            return 0

    def random_ranking(self):
        return self.rng.choice(
            self.average_profile.unique_rankings, size=1, p=self.average_profile.multiplicities
        )[0]

    def random_borda(self):
        return self.rng.choice(
            self.average_profile.unique_bordas, size=1, p=self.average_profile.multiplicities
        )[0]

    def random_profile(self, n):
        return self._random_profile_using_random_borda(n=n)

    @cached_property
    def average_profile(self):
        return Profile.from_d_borda_multiplicity({
            borda: multiplicity / self.base_profile.n
            for borda, multiplicity in self.base_profile.d_borda_multiplicity.items()
        })
