from math import prod

import numpy as np
import sympy
from more_itertools import powerset
from scipy.integrate import nquad
from scipy.optimize import minimize

from actinvoting.util_cache import cached_property


class WorkSession:
    """
    A working session specifying the culture, the candidate of interest and the vector of threshold alpha.

    This is used to study the probability that candidate `c` is an alpha-winner in a profile drawn from the culture.

    Parameters
    ----------
    culture : Culture
        The culture, i.e., the probability distribution over the rankings.
    c : int
        The candidate of interest.
    alpha : list of sympy.Rational, optional
        The vector of thresholds alpha. By convention, it is of size `m` and the coefficient alpha[c] is not used.
        If alpha is not specified, it is set to [1/2, 1/2, ..., 1/2], corresponding to the usual notion of Condorcet
        winner.
    tau : list of sympy.Rational, optional
        The log saddle point. If not specified, it is computed by maximizing the psi function. The advantage of
        specifying it is that it can be used for formal computations (whereas when it is computed by the work session
        itself, this is done by numerical optimization).
    """


    def __init__(self, culture, c, alpha=None, tau=None):
        if alpha is None:
            alpha = [sympy.Rational(1, 2)] * culture.m
        self.culture = culture
        self.m = self.culture.m
        self.c = c
        self.candidates = set(range(self.m))
        self.adversaries = self.candidates - {self.c}
        self.alpha = alpha
        self.beta = [1 - alpha[d] for d in range(self.m)]
        self.beta_short = self.beta[:self.c] + self.beta[self.c + 1:]
        self.x = sympy.symarray("x", self.m)
        self.t = sympy.symarray("t", self.m)
        self._tau = tau

    @cached_property
    def characteristic_polynomial(self):
        """
        The characteristic polynomial P of the culture.

        Returns
        -------
        sympy.Expr
            The characteristic polynomial.
        """
        return sympy.Add(*[
            self.culture.proba_high_low(self.c, set(higher), self.adversaries - set(higher))
            * sympy.Mul(*[self.x[d] for d in higher])
            for higher in powerset(self.adversaries)
        ])

    @cached_property
    def cumulant(self):
        """
        The cumulant generating function K.

        Returns
        -------
        sympy.Expr
            The cumulant generating function K(t) = log(P(e^t)).
        """
        return sympy.log(self.characteristic_polynomial.subs({self.x[j]: sympy.exp(self.t[j]) for j in range(self.m)}))

    @cached_property
    def psi(self):
        """
        The psi function.

        Returns
        -------
        sympy.Expr
            The psi function, i.e., psi(t) = -K(t) + beta^T t.
        """
        return - self.cumulant + sum([self.beta[d] * self.t[d] for d in self.adversaries])

    @cached_property
    def tau(self):
        """
        The log saddle point.

        In the paper, tau is of size m-1, but here we return the full m-size vector. We set conventionally tau[c] = 0.

        Returns
        -------
        list of sympy.Rational
            The log saddle point, i.e., the unique argmax of psi.
        """
        if self._tau is not None:
            return self._tau
        psi_lambdified = sympy.lambdify([*self.t[:self.c], *self.t[self.c + 1:]], self.psi, "numpy")

        def minus_psi_vector_input(v):
            return -psi_lambdified(*v)
        res = minimize(minus_psi_vector_input, [0.0] * (self.m - 1))
        tau_short = res.x
        tau_long = list(tau_short[:self.c]) + [0.] + list(tau_short[self.c:])
        return tau_long

    @cached_property
    def zeta(self):
        """
        The saddle point.

        In the paper, zeta is of size m-1, but here we return the full m-size vector. We set conventionally zeta[c] = 1.

        Returns
        -------
        list of sympy.Rational
            The saddle point, i.e., e^tau.
        """
        # In the paper, zeta is of size m-1, but here we return the full m-size vector.
        # We set conventionally zeta[c] = 1.
        return [sympy.exp(tau_j) for tau_j in self.tau]

    @cached_property
    def p_of_zeta(self):
        """
        The value of the characteristic polynomial at the saddle point.

        Returns
        -------
        sympy.Expr
            The value of the characteristic polynomial at the saddle point.
        """
        return self.characteristic_polynomial.subs({self.x[j]: self.zeta[j] for j in range(self.m)})

    @cached_property
    def hessian_of_p_at_zeta(self):
        """
        The Hessian of the characteristic polynomial at the saddle point.

        Returns
        -------
        sympy.Matrix
            The Hessian of the characteristic polynomial at the saddle point.
        """
        return sympy.hessian(
            self.characteristic_polynomial, [*self.x[:self.c], *self.x[self.c + 1:]]
        ).subs({self.x[j]: self.zeta[j] for j in range(self.m)})

    @cached_property
    def hessian_of_k_at_tau(self):
        """
        The Hessian of the cumulant at the log saddle point.

        Returns
        -------
        sympy.Matrix
            The Hessian of the cumulant at the log saddle point.
        """
        return sympy.hessian(
            self.cumulant, [*self.t[:self.c], *self.t[self.c + 1:]]
        ).subs({self.t[j]: self.tau[j] for j in range(self.m)})

    @cached_property
    def hessian_of_k_at_tau_computed_from_h_p_zeta(self):
        """
        The Hessian of the cumulant at the log saddle point, computed from the Hessian of the characteristic polynomial.

        Returns
        -------
        sympy.Matrix
            The Hessian of the cumulant at the log saddle point, computed from the Hessian of the characteristic
            polynomial. This must equal to hessian_of_k_at_tau. It can be used as a sanity check. Furthermore, due
            to the symbolic computation via sympy, it may happen that one expression is simpler than the other.
        """
        p_zeta = self.p_of_zeta
        h_p_zeta = self.hessian_of_p_at_zeta
        beta_diagonal = np.diag(self.beta_short)
        zeta_short = list(self.zeta)[:self.c] + list(self.zeta)[self.c + 1:]
        zeta_diagonal = np.diag(zeta_short)
        return zeta_diagonal @ h_p_zeta @ zeta_diagonal / p_zeta + beta_diagonal - np.outer(self.beta_short, self.beta_short)

    @cached_property
    def det_hessian_of_k_at_tau(self):
        """
        The determinant of the Hessian of the cumulant at the log saddle point.

        Returns
        -------
        sympy.Expr
            The determinant of the Hessian of the cumulant at the log saddle point.
        """
        return sympy.det(self.hessian_of_k_at_tau)

    @cached_property
    def inverse_of_hessian_of_k_at_tau(self):
        """
        The inverse of the Hessian of the cumulant at the log saddle point.

        Returns
        -------
        sympy.Matrix
            The inverse of the Hessian of the cumulant at the log saddle point.
        """
        return self.hessian_of_k_at_tau.inv()

    @cached_property
    def subcritical_candidates(self):
        """
        The subcritical candidates.

        In practice, we compute the subcritical candidates as those for which tau[j] is less than 0, but not
        numerically too close to 0 (cf. critical candidates).

        Returns
        -------
        set of int
            The subcritical candidates.
        """
        return {j for j in range(self.m) if j != self.c and self.tau[j] < 0 and not np.isclose(self.tau[j], 0)}

    @cached_property
    def critical_candidates(self):
        """
        The critical candidates.

        In practice, we compute the critical candidates as those for which tau[j] is numerically close enough to 0.

        Returns
        -------
        set of int
            The critical candidates.
        """
        return {j for j in range(self.m) if j != self.c and np.isclose(self.tau[j], 0)}

    @cached_property
    def n_subcritical_candidates(self):
        """
        The number of subcritical candidates.

        Returns
        -------
        int
            The number of subcritical candidates.
        """
        return len(self.subcritical_candidates)

    @cached_property
    def n_critical_candidates(self):
        """
        The number of critical candidates.

        Returns
        -------
        int
            The number of critical candidates.
        """
        return len(self.critical_candidates)

    @cached_property
    def matrix_m(self):
        """
        The matrix M.

        This is the submatrix of the inverse of the Hessian of the cumulant at the log saddle point corresponding to the
        critical candidates. It appears in the theoretical equivalent.

        Returns
        -------
        np.ndarray
            The matrix M.
        """
        critical = list(self.critical_candidates)
        return self.inverse_of_hessian_of_k_at_tau[critical, :][:, critical]

    @cached_property
    def integral_of_gaussian_m_with_error(self):
        """
        The integral appearing in the theoretical equivalent, along the margin of error in the numerical integration.

        Returns
        -------
        tuple of float
            The integral along with the margin of error.
        """
        if self.n_critical_candidates == 0:
            return 1, 0
        m = np.array(self.matrix_m, dtype=float)
        def f(*u):
            return np.exp(- np.array(u) @ m @ np.array(u) / 2)
        return nquad(f, [[0, np.inf]] * self.n_critical_candidates)

    @cached_property
    def integral_of_gaussian_m(self):
        """
        The integral appearing in the theoretical equivalent.

        Returns
        -------
        float
            The integral.
        """
        return self.integral_of_gaussian_m_with_error[0]

    def equivalent(self, n):
        """
        The equivalent of the probability that candidate c is an alpha-winner in a profile of size n.

        Parameters
        ----------
        n: int
            The number of voters.

        Returns
        -------
        float
            The value of the theoretical equivalent of the probability.
        """
        if self.n_subcritical_candidates + self.n_critical_candidates < self.m - 1:
            raise NotImplementedError("Supercritical cases are not implemented yet.")
        if self.n_subcritical_candidates == self.m - 1:
            numerator = self.p_of_zeta ** n
            denominator = prod(
                [(1 - self.zeta[j]) * self.zeta[j] ** (np.ceil(self.beta[j] * n) - 1) for j in
                 self.adversaries]
            ) * sympy.sqrt(
                (2 * np.pi * n) ** (self.m - 1) * self.det_hessian_of_k_at_tau
            )
            return numerator / denominator
        numerator = self.p_of_zeta**n * self.integral_of_gaussian_m
        denominator = prod(
            [(1 - self.zeta[j]) * self.zeta[j]**(np.ceil(self.beta[j] * n) - 1) for j in self.subcritical_candidates]
        ) * sympy.sqrt(
            (2 * np.pi)**(self.m - 1) * n**self.n_subcritical_candidates * self.det_hessian_of_k_at_tau
        )
        return numerator / denominator

    def exact_probability(self, n):
        """
        The exact probability that candidate c is an alpha-winner in a profile of size n.

        Take self.polynom_of_duels, raise to the power n, then take the coefficients of all the monomials such
        that for each adversary j, the exponent of x[j] is less than beta[j] n.

        Be careful, this is computationally expensive.

        Parameters
        ----------
        n: int
            The number of voters.

        Returns
        -------
        float
            The exact probability.
        """
        p_n = self.characteristic_polynomial ** n
        probability = sympy.Rational(0, 1)
        for monomial in sympy.expand(p_n).as_ordered_terms():
            exponents = monomial.as_powers_dict()
            if all(exponents[self.x[j]] < n * self.beta[j] for j in self.adversaries):
                probability += monomial.subs({self.x[j]: 1 for j in self.adversaries})
        return probability
