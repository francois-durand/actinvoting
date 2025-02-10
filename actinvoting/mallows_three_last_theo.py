import numpy as np


def mallows_three_last_theo(n, rho):
    """
    Compute the theoretical equivalent of P(3 is CW) in "M 3 last"

    "M 3 last" is a Mallows model with pole [1, 2, 3].

    Parameters
    ----------
    n: int
        Number of voters.
    rho: float
        Concentration parameter for the Mallows model.

    Returns
    -------
    float
        Theoretical equivalent of the probability that candidate 3 is the Condorcet winner in "M 3 last".
    """
    zeta = np.array([np.exp(-3*rho/2), np.exp(-rho/2)])
    gamma = 1 / ((1 + np.exp(-rho)) * (1 + np.exp(-rho) + np.exp(-2 * rho)))
    p_zeta = 2 * gamma * np.exp(-2*rho) * (1 + np.exp(-rho/2) + np.exp(-rho))
    det = (1/4) * np.exp(-rho/2) * (1 + np.exp(-rho)) / (1 + np.exp(-rho/2) + np.exp(-rho))**2
    the_product = (1 - zeta[0]) * zeta[0]**(np.ceil(n/2) - 1) * (1 - zeta[1]) * zeta[1]**(np.ceil(n/2) - 1)
    denominator = the_product * 2 * np.pi * n * np.sqrt(det)
    return p_zeta**n / denominator
