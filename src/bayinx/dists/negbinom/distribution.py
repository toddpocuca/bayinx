from typing import Optional

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.types import ArrayObject

from .pars import MeanInvOverdispNegBinom


class NegBinom(Distribution):
    """
    A Negative Binomial distribution.

    Parameters:
        mu: Parameterizes a Negative Binomial distribution by its mean. E(X) = mu.
        theta: Parameterizes a Negative Binomial distribution by its inverse overdispersion. Var(X) = mu + mu^2/theta.
    """

    par: Parameterization

    def __init__(
        self,
        mu: Optional[ArrayObject] = None,
        theta: Optional[ArrayObject] = None,
    ):
        if mu is not None and theta is not None:
            self.par = MeanInvOverdispNegBinom(mu, theta)
        else:
            raise TypeError(f"Expected mu: {mu}, and theta: {theta} to be not None.")
