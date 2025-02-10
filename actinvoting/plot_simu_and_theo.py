import matplotlib.pyplot as plt
import numpy as np

from actinvoting.my_tikzplotlib_save import my_tikzplotlib_save


def plot_simu_and_theo(
    equivalent_probas=None, monte_carlo_probas=None, exact_probas=None,
    ns_equivalent_probas=None, ns_monte_carlo_probas=None, ns_exact_probas=None,
    label_equivalent="Theoretical equivalent", label_monte_carlo="Monte-Carlo results", label_exact="Exact results",
    x_label="Number of voters $n$", y_label=r"$\mathbb{P}(m \text{ is CW})$",
    log_scale=False, xmax=None, legend_loc=None, verbose=True, file_name=None
):
    """
    Plot the theoretical, Monte-Carlo and exact probabilities of the Condorcet winner as functions of n.

    Parameters
    ----------
    equivalent_probas: list of floats
        Values of the theoretical equivalent for each n in `ns_equivalent_probas`.
    monte_carlo_probas: list of floats
        Values of the Monte-Carlo estimates for each n in `ns_monte_carlo_probas`.
    exact_probas: list of floats
        Values of the exact probabilities for each n in `ns_exact_probas`.
    ns_equivalent_probas: list of ints
        Values of n for which the theoretical equivalent has been computed.
    ns_monte_carlo_probas: list of ints
        Values of n for which the Monte-Carlo estimates have been computed.
    ns_exact_probas: list of ints
        Values of n for which the exact probabilities have been computed.
    label_equivalent: str
        Label for the theoretical equivalent in the plot.
    label_monte_carlo: str
        Label for the Monte-Carlo estimates in the plot.
    label_exact: str
        Label for the exact probabilities in the plot.
    x_label: str
        Label for the x-axis.
    y_label: str
        Label for the y-axis.
    log_scale: bool
        Whether to use a log scale for the y-axis.
    xmax: float
        Maximum value for the x-axis. If specified, the minimal value is set to 0.
    legend_loc
        The location of the legend.
    verbose
        Whether to print the data.
    file_name
        If not None, the plot is saved in a file with this name, using tikzplotlib.
    """
    if equivalent_probas is not None:
        plt.plot(ns_equivalent_probas, equivalent_probas, label=label_equivalent)
        d_n_equivalent_proba = {k: v for k, v in zip(ns_equivalent_probas, equivalent_probas)}
        if verbose:
            print(f"{d_n_equivalent_proba=}")

    if exact_probas is not None:
        plt.plot(ns_exact_probas, exact_probas, label=label_exact)
        d_n_exact_proba = {k: v for k, v in zip(ns_exact_probas, exact_probas)}
        if verbose:
            print(f"{d_n_exact_proba=}")

    if monte_carlo_probas is not None:
        if log_scale:
            indices = np.where(np.array(monte_carlo_probas)  > 0)[0]
        else:
            indices = np.array(range(len(ns_monte_carlo_probas)))
        plt.plot(np.array(ns_monte_carlo_probas)[indices], np.array(monte_carlo_probas)[indices],
                 label=label_monte_carlo)
        d_n_mc_proba = {k: v for k, v in zip(ns_monte_carlo_probas, monte_carlo_probas)}
        if verbose:
            print(f"{d_n_mc_proba=}")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if log_scale:
        plt.yscale("log")
    else:
        if xmax is not None:
            plt.ylim(0., xmax)
    plt.legend(loc=legend_loc)
    if file_name is not None:
        my_tikzplotlib_save(file_name)
