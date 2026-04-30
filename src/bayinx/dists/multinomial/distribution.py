
from jaxtyping import ArrayLike

from bayinx.core.distribution import Distribution, Parameterization

from .pars import LogProbsMultinomial, ProbsMultinomial


class Multinomial(Distribution):
    """
    A Multinomial distribution.
    """

    par: Parameterization

    def __init__(
        self,
        n: ArrayLike,
        probs: None | ArrayLike = None,
        logprobs: None | ArrayLike = None
    ):
        if probs is not None:
            self.par = ProbsMultinomial(n, probs)
        elif logprobs is not None:
            self.par = LogProbsMultinomial(n, logprobs)
        else:
            raise TypeError("Expected 'probs' or 'logprobs' to be not None.")
