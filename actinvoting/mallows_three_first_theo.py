import numpy as np


def mallows_three_first_theo(n, rho):
    """
    Compute the theoretical equivalent of 1 - P(3 is CW) in "M 3 first"

    "M 3 first" is a Mallows model with pole [3, 2, 1].

    Parameters
    ----------
    n: int
        Number of voters.
    rho: float
        Concentration parameter for the Mallows model.

    Returns
    -------
    float
        Theoretical equivalent of the probability that candidate 3 fails to be the Condorcet winner in "M 3 first".
    """
    constant = np.sqrt(2 / (np.pi * n))
    numerator = ((2 * np.exp(-rho)) / (1 + np.exp(-rho)))**n
    denominator = (1 - np.exp(-rho)) * np.exp(-rho * (np.floor(n / 2)))
    return constant * numerator / denominator
