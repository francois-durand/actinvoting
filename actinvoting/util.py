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
        >>> list(borda_from_ranking([0, 1, 2, 3, 4, 5]))
        [5, 4, 3, 2, 1, 0]
        >>> list(borda_from_ranking([2, 5, 0, 1, 3, 4]))
        [3, 2, 5, 1, 0, 4]
    """
    m = len(ranking)
    return m - 1 - np.argsort(ranking)


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
        >>> list(ranking_from_borda([5, 4, 3, 2, 1, 0]))
        [0, 1, 2, 3, 4, 5]
        >>> list(ranking_from_borda([3, 2, 5, 1, 0, 4]))
        [2, 5, 0, 1, 3, 4]

    """
    return np.argsort(-np.array(borda))


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
        0
        >>> kendall_tau_id_ranking([1, 0, 2, 3, 4, 5])
        1
        >>> kendall_tau_id_ranking([2, 5, 0, 1, 3, 4])
        6
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
        0
        >>> kendall_tau_id_borda([4, 5, 3, 2, 1, 0])
        1
        >>> kendall_tau_id_borda([3, 2, 5, 1, 0, 4])
        6
    """
    borda = np.array(borda)
    m = len(borda)
    return (borda[:, np.newaxis] < borda[np.newaxis, :])[np.triu_indices(m)].sum()
