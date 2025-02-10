import numpy as np


def borda_from_ranking(ranking):
    """
    Deduce a Borda vector from a ranking.

    Parameters
    ----------
    ranking: List
        A ranking.

    Returns
    -------
    ndarray
        The same ranking, in Borda format.

    Examples
    --------
        >>> borda_from_ranking([0, 1, 2, 3, 4, 5])
        array([5, 4, 3, 2, 1, 0])
        >>> borda_from_ranking([2, 5, 0, 1, 3, 4])
        array([3, 2, 5, 1, 0, 4])
    """
    m = len(ranking)
    borda = np.zeros(m, int)
    borda[np.array(ranking)] = np.arange(m - 1, -1, -1)
    return borda


def ranking_from_borda(borda):
    """
    Deduce ranking from Borda vector.

    Parameters
    ----------
    borda: List
        A ranking in Borda format.

    Returns
    -------
    ndarray
        The same ranking, in ranking format.

    Examples
    --------
        >>> ranking_from_borda([5, 4, 3, 2, 1, 0])
        array([0, 1, 2, 3, 4, 5])
        >>> ranking_from_borda([3, 2, 5, 1, 0, 4])
        array([2, 5, 0, 1, 3, 4])
    """
    m = len(borda)
    ranking = np.zeros(m, int)
    ranking[m - 1 - np.array(borda)] = np.arange(m)
    return ranking


def kendall_tau_id_ranking(ranking):
    """
    Kendall-tau distance between a ranking and the identity ranking [0, 1, ..., m-1].

    Parameters
    ----------
    ranking: List
        A ranking.

    Returns
    -------
    int
        The Kendall-tau distance (swap distance) between the input and the identity ranking.

    Examples
    --------
        >>> kendall_tau_id_ranking([0, 1, 2, 3, 4, 5])
        np.int64(0)
        >>> kendall_tau_id_ranking([1, 0, 2, 3, 4, 5])
        np.int64(1)
        >>> kendall_tau_id_ranking([2, 5, 0, 1, 3, 4])
        np.int64(6)
    """
    ranking = np.array(ranking)
    m = len(ranking)
    return (ranking[:, np.newaxis] > ranking[np.newaxis, :])[np.triu_indices(m)].sum()


def kendall_tau_id_borda(borda):
    """
    Kendall-tau distance between a Borda vector and the identity ranking, of Borda vector [m, m-1, ..., 0].

    Parameters
    ----------
    borda: List
        A ranking in Borda format.

    Returns
    -------
    int
        The Kendall-tau distance (swap distance) between the input and the identity ranking.

    Examples
    --------
        >>> kendall_tau_id_borda([5, 4, 3, 2, 1, 0])
        np.int64(0)
        >>> kendall_tau_id_borda([4, 5, 3, 2, 1, 0])
        np.int64(1)
        >>> kendall_tau_id_borda([3, 2, 5, 1, 0, 4])
        np.int64(6)
    """
    borda = np.array(borda)
    m = len(borda)
    return (borda[:, np.newaxis] < borda[np.newaxis, :])[np.triu_indices(m)].sum()
