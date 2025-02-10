import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from actinvoting.cultures.culture_impartial import CultureImpartial
from actinvoting.probability_monte_carlo import probability_monte_carlo
from actinvoting.work_session import WorkSession


def plot_speed_ic(m, ns, n_samples):
    """
    Check that for IC, the error term is in O(n^{-1/2}).

    Parameters
    ----------
    m: int
        Number of candidates.
    ns: list of int
        List of values for the number of voters n. Typically, this should be used for even values of n.
    n_samples
        Number of samples for the Monte Carlo method.
    """
    culture = CultureImpartial(m=m)
    session = WorkSession(culture=culture, c=m-1)
    theoretical_limit = session.equivalent(n=1)

    def proba_mc(n):
        return probability_monte_carlo(
            factory=lambda : culture.random_profile(n=n),
            n_samples=n_samples,
            test=lambda profile: profile.is_condorcet_winner[profile.m - 1]
        )

    # For each n i ns, compute proba_mc(n). Then plot proba_mc against n
    # Use log scale for both axes
    ys = [theoretical_limit - proba_mc(n) for n in ns]
    plt.plot(ns, ys)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Number of voters")
    plt.ylabel("Difference with the limit")

    # Compute a linear regression of the curve in log scale, in order to have the lead exponent
    # Print this exponent
    res = linregress([np.log(n) for n in ns], [np.log(float(y)) for y in ys])
    slope = res[0]
    print(f"Lead exponent: {slope}")
