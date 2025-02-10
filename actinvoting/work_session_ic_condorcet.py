import numpy as np
import sympy
from scipy.integrate import nquad

from actinvoting.cultures.culture_impartial import CultureImpartial
from actinvoting.util_cache import cached_property
from actinvoting.work_session import WorkSession


class WorkSessionICCondorcet(WorkSession):
    """
    A work session dedicated to Impartial Culture, implementing the second term in the asymptotic expansion.
    """

    def __init__(self, m):
        culture = CultureImpartial(m=m)
        super().__init__(culture=culture, c=m-1, tau=[sympy.Rational(0, 1)] * m)

    @cached_property
    def subcritical_candidates(self):
        # No candidate is subcritical.
        return set()

    @cached_property
    def critical_candidates(self):
        # All candidates are critical.
        return self.adversaries

    @cached_property
    def integral_for_error_term_with_error(self):
        m = np.array(self.matrix_m, dtype=float)
        def f(*u):
            return np.array(u).sum() * np.exp(- np.array(u) @ m @ np.array(u) / 2)
        return nquad(f, [[0, np.inf]] * self.n_critical_candidates)

    @cached_property
    def integral_for_error_term(self):
        return self.integral_for_error_term_with_error[0]

    def asymptotics(self, n):
        """
        The beginning of the asymptotic expansion: limit + term in n^{-1/2}.

        Parameters
        ----------
        n: int
            The number of voters

        Returns
        -------
        float
            The estimated probability, using the first two terms of the asymptotic expansion.
        """
        limit = self.equivalent(n=n)
        if n % 2 == 1:
            return limit
        else:
            a_1 = - 6 * self.integral_for_error_term / (
                (self.m + 1) * np.sqrt((2 * np.pi)**(self.m - 1) * float(self.det_hessian_of_k_at_tau))
            )
            return limit + a_1 / np.sqrt(n)
