from typing import Optional

from jaxtyping import Array, ArrayLike, Real

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.node import Node

from .pars import LogRatePoisson, RatePoisson


class Poisson(Distribution):
    """
    A Poisson distribution.
    """

    par: Parameterization


    def __init__(
        self,
        rate: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        log_rate: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None
    ):
        if rate is not None:
            self.par = RatePoisson(rate)
        elif log_rate is not None:
            self.par = LogRatePoisson(log_rate)
        else:
            raise TypeError(f"Expected rate: {rate} or log_rate: {log_rate} to be not None.")
