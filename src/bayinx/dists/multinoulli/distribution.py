from typing import Optional

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.types import ArrayObject

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
        probs: Optional[ArrayObject] = None,
        logprobs: Optional[ArrayObject] = None
    ):
        if probs is not None:
            self.par = ProbsMultinoulli(probs)
        elif logprobs is not None:
            self.par = LogProbsMultinoulli(logprobs)
        else:
            raise TypeError("Expected 'probs' or 'logprobs' to be not None.")
