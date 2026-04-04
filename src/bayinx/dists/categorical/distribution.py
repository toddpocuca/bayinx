from typing import Optional

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.types import ArrayObject

from .pars import LogProbsCategorical, ProbsCategorical


class Categorical(Distribution):
    """
    A Categorical distribution.
    """

    par: Parameterization

    def __init__(
        self,
        probs: Optional[ArrayObject] = None,
        logprobs: Optional[ArrayObject] = None
    ):
        if probs is not None:
            self.par = ProbsCategorical(probs)
        elif logprobs is not None:
            self.par = LogProbsCategorical(logprobs)
        else:
            raise TypeError("Expected 'probs' or 'logprobs' to be not None.")
