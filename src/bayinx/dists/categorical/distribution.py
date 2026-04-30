
from jaxtyping import ArrayLike

from bayinx.core.distribution import Distribution, Parameterization

from .pars import LogProbsCategorical, ProbsCategorical


class Categorical(Distribution):
    """
    A Categorical distribution.
    """

    par: Parameterization

    def __init__(
        self,
        probs: None | ArrayLike = None,
        logprobs: None | ArrayLike = None
    ):
        if probs is not None:
            self.par = ProbsCategorical(probs)
        elif logprobs is not None:
            self.par = LogProbsCategorical(logprobs)
        else:
            raise TypeError("Expected 'probs' or 'logprobs' to be not None.")
