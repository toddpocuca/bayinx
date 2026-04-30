
from jaxtyping import ArrayLike

from bayinx.core.distribution import Distribution, Parameterization

from .pars import LogRatePoisson, RatePoisson


class Poisson(Distribution):
    """
    A Poisson distribution.

    Parameters:
        rate: Parameterizes a Poisson distribution by its rate.
        log_rate: Parameterizes a Poisson distribution by the log-transformed rate.
    """

    par: Parameterization


    def __init__(
        self,
        rate: None | ArrayLike = None,
        log_rate: None | ArrayLike = None
    ):
        if rate is not None:
            self.par = RatePoisson(rate)
        elif log_rate is not None:
            self.par = LogRatePoisson(log_rate)
        else:
            raise TypeError(f"Expected rate: {rate} or log_rate: {log_rate} to be not None.")
