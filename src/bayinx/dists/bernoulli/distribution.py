
from jaxtyping import ArrayLike

from bayinx.core.distribution import Distribution, Parameterization

from .pars import (
    LogitProbFailureBernoulli,
    LogitProbSuccessBernoulli,
    ProbFailureBernoulli,
    ProbSuccessBernoulli,
)


class Bernoulli(Distribution):
    """
    A Bernoulli distribution.
    """

    par: Parameterization


    def __init__(
        self,
        p: None | ArrayLike = None,
        q: None | ArrayLike = None,
        logit_p: None | ArrayLike = None,
        logit_q: None | ArrayLike = None
    ):
        """
        Construct a Bernoulli distribution by selecting a parameterization.

        Parameters:
            p: Parameterize a Bernoulli distribution by its probability of success.
            q: Parameterize a Bernoulli distribution by its probability of failure.
            logit_p: Parameterize a Bernoulli distribution by its logit probability of success.
            logit_q: Parameterize a Bernoulli distribution by its logit probability of failure.
        """
        if p is not None:
            self.par = ProbSuccessBernoulli(p)
        elif q is not None:
            self.par = ProbFailureBernoulli(q)
        elif logit_p is not None:
            self.par = LogitProbSuccessBernoulli(logit_p)
        elif logit_q is not None:
            self.par = LogitProbFailureBernoulli(logit_q)
        else:
            raise TypeError(f"Expected at least one of p: {p}, q: {q}, logit_p: {logit_p}, logit_q: {logit_q} to be not None.")
