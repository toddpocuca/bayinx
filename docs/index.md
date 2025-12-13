# Getting Started with Bayinx

Welcome to Bayinx (Bayesian inference with JAX). This guide will help you install the package and run your first model.

## Installation

Bayinx requires JAX and a few extra libraries in the JAX ecosystem (Equinox, Diffrax, etc). The easiest way to get started is by installing from PyPi using your favourite python package manager. For example, with `uv` you can install Bayinx like so:

```bash
# Ensure you are in your project environment
uv add bayinx
```

This installs the bare-bones version of Bayinx, however if you need additional functionality like GPU support or support for post-processing of fitted models, there are a couple of dependency groups:
```bash
# Ensure you are in your project environment
uv add 'bayinx[cuda]' # Installs Bayinx with CUDA support
```

## Defining Models In Bayinx

You can now get started!

Models are defined by constructing a class that inherits from the `Model` base class. For example, we can define a simple model that describes fitting a Normal distribution to a collection of observations:

```py
from bayinx.dists import Normal, Exponential
from bayinx.nodes import Continuous, Observed
from bayinx import Model, define
from jaxtyping import Array

class SimpleNormalModel(Model):
    mean: Continuous[Array] = define(shape = ())
    std: Continuous[Array] = define(shape = (), lower = 0)

    x: Observed[Array] = define(shape = 'n_obs')

    def model(self, target):
        # Accumulate likelihood
        self.x << Normal(self.mean, self.std)

        return target
```

> If you're coming from Stan this will look largely familiar, the data and parameters blocks have been combined into the attribute definitions, while the model block is defined in the `model` method, and distribution statements are written using `<<` instead of `~`.


## Fitting Models With Bayinx
Bayinx uses Variational Inference with Normalizing Flows (NF) to approximate the posterior distribution, where the NF architecture can be customized to your preference. We'll simulate some data for demonstration:

```py
import jax.random as jr

n_obs = 100
true_mean = 10.0
true_std = 5.0

# Simulate data
x_data = jr.normal(jr.key(0), (n_obs, )) * true_std + true_mean
```

The approximation to the posterior can then be created with the `Posterior` class and optimized further down the line:

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
posterior.fit() # Optimize the approximation
```

Once fitted, you can sample from the approximated posterior distribution to get Monte Carlo estimates for your parameters:

```py
# Sample the posterior distribution for 'mean'
mean_draws = posterior.sample('mean', int(1e6))

print(f"Analytic Posterior Mean for 'mean': {x_data.mean():.4f}")
print(f"Posterior Mean Estimate for 'mean': {mean_draws.mean():.4f}")
```
```
Analytic Posterior Mean for 'mean': 10.5465
Posterior Mean Estimate for 'mean': 10.5455
```
