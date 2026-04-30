# Getting Started with Bayinx

Welcome to Bayinx (Bayesian inference with JAX), a probabilistic programming language embedded in Python.
This guide will help you install the package and quickly overview how Bayinx works, but if you would like a more thorough tutorial check out [Basic Usage](tutorials/basic.md) for those of you unfamiliar to probabilistic programming or [Coming From Stan](tutorials/stan.md) for Stan users.
If you are unfamiliar with Python check out the [official Python tutorial](https://docs.python.org/3/tutorial/index.html) or another resource to get up to speed with programming in Python.

## Installation

Bayinx requires JAX and a few extra libraries in the JAX ecosystem.
The easiest way to get started is by installing from PyPi using your favourite python package manager:

### [`uv`](https://docs.astral.sh/uv/)

```bash
# Ensure you are in your project environment
uv add bayinx
```

This installs the bare-bones version of Bayinx, however if you need additional functionality like GPU support, there are a couple of dependency groups:
```bash
# Ensure you are in your project environment
uv add 'bayinx[cuda]' # Installs Bayinx with CUDA support
```

### [`pip`](https://pypi.org/project/pip/)

```bash
# Ensure you are in your project environment
pip install bayinx
```

This installs the bare-bones version of Bayinx, however if you need additional functionality like GPU support, there are a couple of dependency groups:
```bash
# Ensure you are in your project environment
pip install 'bayinx[cuda]' # Installs Bayinx with CUDA support
```

## Defining Models In Bayinx

You can now get started!

Models are defined by writing a class that inherits from the `Model` base class.
For example, we can define a simple model that describes a collection of observations derived from a Normal distribution:

```py
import jax.random as jr

from bayinx.dists import Normal, Exponential
from bayinx import Model, observed, stochastic
from jaxtyping import Scalar, Array

class SimpleNormalModel(Model):
    mu: Scalar = stochastic(shape = ())
    std: Array = stochastic(shape = (), lower = 0)

    x: Array = observed(shape = 'n_obs')

    def model(self, target):
        # Accumulate likelihood
        self.x << Normal(self.mu, self.std)

# Simulate fake data
n_obs = 30
true_mu = 10.0
true_std = 3.0

# Simulate data
x_data = jr.normal(jr.key(0), (n_obs, )) * true_std + true_mu
```

Parameters are attributes with the `stochastic` descriptor, while any data is marked with the `observed` descriptor.
Additional metadata for an attribute is passed as arguments to these descriptors, for example by assigning shapes `define(shape = ...)` or a constraint `define(lower = ..., upper = ...)`.

## Fitting Models With Bayinx
Bayinx uses variational inference with [normalizing flows](nf.md) (NFs) to approximate the posterior distribution, where the NF architecture can be customized to your preference.
We'll simulate some data for demonstration:

The approximation to the posterior can then be created with the `Posterior` class and optimized later:

```py
from bayinx import Posterior
from bayinx.flows import DiagAffine

# Construct approximation
posterior = Posterior(
    SimpleNormalModel,
    n_obs = n_obs,
    x = x_data
)
posterior.configure(flowspecs = [DiagAffine()]) # Configure the NF architecture
posterior.fit(stl = True) # Optimize the approximation
```

Once fitted, you can sample from the approximated posterior distribution to get Monte Carlo estimates for your parameters:

```py
# Sample the posterior distribution for 'mean'
mu_draws = posterior.sample('mu', int(5e6), batch_size = int(1e4))

print(f"Analytic Posterior Mean for 'mu': {x_data.mean():.4f}")
print(f"Posterior Mean Estimate for 'mu': {mu_draws.mean():.4f} ± {mu_draws.std() / 5e6**0.5:.4f}")
```
```
Analytic Posterior Mean for 'mu': 10.6796
Posterior Mean Estimate for 'mu': 10.6792 ± 0.0003
```
