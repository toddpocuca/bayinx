
from jaxtyping import ArrayLike

from bayinx.core.distribution import Distribution, Parameterization

from .pars import LogProbsMultinoulli, ProbsMultinoulli


class Multinoulli(Distribution):
    """
    A Multinoulli distribution.

    Parameters:
        probs: Parameterizes a Multinoulli distribution by its probabilities.
        logprobs: Parameterizes a Multinoulli distribution by its log-probabilities.
    """

    par: Parameterization

    def __init__(
        self,
        probs: None | ArrayLike = None,
        logprobs: None | ArrayLike = None
    ):
        if probs is not None:
            self.par = ProbsMultinoulli(probs)
        elif logprobs is not None:
            self.par = LogProbsMultinoulli(logprobs)
        else:
            raise TypeError("Expected 'probs' or 'logprobs' to be not None.")
