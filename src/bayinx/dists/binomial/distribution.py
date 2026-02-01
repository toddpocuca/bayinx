from typing import Optional

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.types import ArrayObject

from .pars import (
    LogitProbFailureBinomial,
    LogitProbSuccessBinomial,
    ProbFailureBinomial,
    ProbSuccessBinomial,
)


class Binomial(Distribution):
    """
    A Binomial distribution.
    """

    par: Parameterization


    def __init__(
        self,
        n: ArrayObject,
        p: Optional[ArrayObject] = None,
        q: Optional[ArrayObject] = None,
        logit_p: Optional[ArrayObject] = None,
        logit_q: Optional[ArrayObject] = None
    ):
        """
        Construct a Binomial distribution by selecting a parameterization.

        Parameters:
            n: Parameterize a Binomial distribution by its total number of trials.
            p: Parameterize a Binomial distribution by its probability of success.
            q: Parameterize a Binomial distribution by its probability of failure.
            logit_p: Parameterize a Binomial distribution by its logit probability of success.
            logit_q: Parameterize a Binomial distribution by its logit probability of failure.
        """
        if p is not None:
            self.par = ProbSuccessBinomial(n, p)
        elif q is not None:
            self.par = ProbFailureBinomial(n, q)
        elif logit_p is not None:
            self.par = LogitProbSuccessBinomial(n, logit_p)
        elif logit_q is not None:
            self.par = LogitProbFailureBinomial(n, logit_q)
        else:
            raise TypeError("Expected at least one of 'p', 'q', 'logit_p', 'logit_q' to be not None.")
