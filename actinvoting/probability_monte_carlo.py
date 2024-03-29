import numpy as np


def probability_monte_carlo(factory, n_samples, test, conditional_on=None):
    """Probability that a random `something` meets some given test.

    Parameters
    ----------
    factory : callable or tuple of callable
        This can be:

        * Either a callable that takes no input and that outputs a (random) `something`,
        * Or a tuple of such factories (cf. examples below).
    n_samples : int
        Number of samples.
    test : callable or tuple of callable
        This can be:

        * Either a function that take as input(s) the output(s) of the factory(ies) and that returns a Boolean.
        * Or a tuple of such functions (cf. examples below).
    conditional_on : callable
        A function that take as input(s) the output(s) of the factory(ies) and that returns a Boolean.
        Default: always True.

    Returns
    -------
    float or tuple of float
        This can be:

        * Either the probability that the output(s) generated by `factory` meet(s) `test`, conditional on the fact
          that it meets `conditional_on`, based on a Monte-Carlo estimation of `n_samples` trials.
        * Or a tuple giving this probability for each member of `test`, when `test` is a tuple itself.

    Examples
    --------
    In this basic example with one factory, we estimate the probability that a random float between 0 and 1 is greater
    than .5, conditionally on being greater than .25:

        >>> np.random.seed(0)
        >>> def rand_number():
        ...     return np.random.rand()
        >>> probability_monte_carlo(factory=rand_number, n_samples=1000,
        ...                         test=lambda x: x > .5, conditional_on=lambda x: x > .25)
        0.657

    In this example with a tuple of factories, we estimate the probability that a random 2*2 matrix and a random vector
    of size 2, both with integer coefficients between -10 included and 11 excluded, have a dot product that is null,
    conditionally on not being null themselves:

        >>> np.random.seed(0)
        >>> def rand_matrix():
        ...     return np.random.randint(-10, 11, (2, 2))
        >>> def rand_vector():
        ...     return np.random.randint(-10, 11, 2)
        >>> def test_dot_zero(matrix, vector):
        ...     return np.all(np.dot(matrix, vector) == 0)
        >>> def test_non_trivial(matrix, vector):
        ...     return not np.all(matrix == 0) and not np.all(vector == 0)
        >>> probability_monte_carlo(factory=(rand_matrix, rand_vector), n_samples=10000,
        ...                         test=test_dot_zero, conditional_on=test_non_trivial)
        0.0003

    In the following example, we estimate the probability that a random float between 0 and 1 is greater than .5,
    and the probability that it is greater than .75, conditionally on being greater than .25:

        >>> np.random.seed(0)
        >>> def rand_number():
        ...     return np.random.rand()
        >>> probability_monte_carlo(factory=rand_number, n_samples=1000,
        ...                         test=(lambda x: x > .5, lambda x: x > .75), conditional_on=lambda x: x > .25)
        (0.657, 0.342)

    When using a tuple of tests, the same sample is used to estimate each probability.
    """
    if not isinstance(factory, tuple):
        factory = (factory,)
    is_test_tuple = isinstance(test, tuple)
    if not is_test_tuple:
        test = (test,)
    l_test_success = [0 for _ in test]
    i_samples = 0
    while i_samples < n_samples:
        somethings = [f() for f in factory]
        if conditional_on is None or conditional_on(*somethings):
            i_samples += 1
            for i_test, the_test in enumerate(test):
                if the_test(*somethings):
                    l_test_success[i_test] += 1
    l_test_rate = [successes / n_samples for successes in l_test_success]
    if is_test_tuple:
        return tuple(l_test_rate)
    else:
        return l_test_rate[0]
