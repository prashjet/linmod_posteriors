import numpy as np


class BetaPrior:
    """The prior on beta defined by
    P(beta) \propto exp(-1/2 (beta-mu_0)^2/sig_0^2 ) if F beta + g >= 0
            \propto 0                                   otherwise
    Parameters
    ----------
    p : int
        len(beta)
    sig_0 : float
        prior width (diagonal)
    mu_0 : type
        prior mean
    n_ineq_constraints : int
        number of inequality constraints
    f : (n_ineq_constraints, p) array of floats
        beta coefficients for inequality constraint
    g : (n_ineq_constraints,) array of floats
        values for inequality constraint
    """

    def __init__(self,
                 p=None,
                 sig_0=1,
                 mu_0=0.,
                 n_ineq_constraints=0,
                 F=None,
                 g=None):
        # set values
        self.p = p
        self.sig_0 = sig_0
        self.mu_0 = mu_0
        self.n_ineq_constraints = n_ineq_constraints
        if n_ineq_constraints>0:
            m, p2 = F.shape
            if F.shape!=(n_ineq_constraints,p):
                str = "F must have shape ({0},{1})"
                str = str.format(n_ineq_constraints, p)
                raise ValueError(str)
            else:
                self.F = F
            if g.shape != (n_ineq_constraints,):
                str = "g must have shape ({0},)"
                str = str.format(n_ineq_constraints)
                raise ValueError(str)
            else:
                self.g = g


class UnconstrainedBetaPrior(BetaPrior):

    def __init__(self,
                 p=None,
                 sig_0=1,
                 mu_0=0.):
        super().__init__(p=p,
                         sig_0=sig_0,
                         mu_0=mu_0,
                         n_ineq_constraints=0)


class DiagConstrainedBetaPrior(BetaPrior):
    pass


class IdentityConstrainedBetaPrior(DiagConstrainedBetaPrior):
    pass


class PositiveBetaPrior(IdentityConstrainedBetaPrior):

    def __init__(self,
                 p=None,
                 sig_0=1,
                 mu_0=0.):
        n_ineq_constraints = p
        F = np.diag(np.ones(p))
        g = np.zeros(p)
        super().__init__(p=p,
                         sig_0=sig_0,
                         mu_0=mu_0,
                         n_ineq_constraints=n_ineq_constraints,
                         F=F,
                         g=g)


class SimplexBetaPrior(PositiveBetaPrior):
    """
    placeholder
    """
    def __init__(self,
                 p=None,
                 sig_0=1.,
                 mu_0=0.,
                 sum_beta=1.,
                 idx_exclude=-1):
        super().__init__()
