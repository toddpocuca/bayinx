"""
Bayinx is an embedded probabilistic programming language in Python, powered by
[JAX](https://mc-stan.org/). It is heavily inspired by and aims to have
feature parity with [Stan](https://mc-stan.org/), but extends the types of
objects you can work with and focuses on normalizing flows variational
inference for sampling.
"""

from .core.model import Model as Model
from .core.model import define as define
from .posterior import Posterior as Posterior
