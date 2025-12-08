# Bayinx: <ins>Bay</ins>esian <ins>In</ins>ference with JA<ins>X</ins>

Bayinx is an embedded probabilistic programming language in Python, powered by
[JAX](https://mc-stan.org/). It is heavily inspired by and aims to have
feature parity with [Stan](https://mc-stan.org/), but extends the types of
objects you can work with and focuses on normalizing flows variational
inference for sampling.


## Coming From Stan

There are a few differences between the syntax of Bayinx and Stan.
First, as Bayinx is embedded in Python, model definitions are Pythonic and
rely on you defining a class that inherits from the `Model` base class:

```py
class MyModel(Model, init=False):
    # ...
```

> Note: Users should specify `init=False` to avoid static type checkers from
raising irrelevant errors, but more importantly it should remind you that
you should **NOT** implement your own `__init__` method!

The `data` and `parameters` blocks in Stan are then combined into the attribute
definitions with Bayinx. For example, if we are modelling a simple normal distribution
with an unknown mean and variance 1, then we might write:

```py
class MyModel(Model, init=False):
    mean: Continuous[Array] = define(shape = ()) # a scalar mean parameter
    x: Observed[Array] = define(shape = 'n_obs') # a vector of observed values

    # ...
```

The `model` block in Stan is then defined by implementing the `model` method with Bayinx:

```py
class MyModel(Model, init=False):
    mean: Continuous[Array] = define(shape = ())
    x: Observed[Array] = define(shape = 'n_obs')

    def model(self, target):
        # Equivalent to 'x ~ normal(mean, 1.0)' in Stan
        self.x << Normal(self.mean, 1.0)

        return target
```

Notice that the `~` operator in Stan has been replaced with `<<`, and to reference nodes of a model you must work with `self`.

> Note: Bayinx does not currently have something similar to `transformed data` or `transformed parameters`, however that is likely to be included in a future release.

You can then construct the variational approximation to the posterior:

```py
import bayinx as byx
from bayinx.flows import DiagAffine
import jax.numpy as jnp

# Fit variational approximation
posterior = byx.Posterior(MyModel, n_obs = 3, x = jnp.array([-1.0, 0.0, 1.0]))
posterior.configure(flowspecs = [DiagAffine()])
posterior.fit()
```

This approximation can then be worked with by sampling nodes:

```py
mean_draws = posterior.sample('mean', 10000)
print(mean_draws.mean())
```


## Roadmap
- [ ] Implement OT-Flow: https://arxiv.org/abs/2006.00104
- [ ] Allow shape definitions to include expressions (e.g., shape = 'n_obs + 1' will evaluate to the correct specification)
- [ ] Figure out how to dynamically construct distributions such that parameterizations don't require calling new functions, just defining `Exponential(rate ...)` vs. `Exponential(scale = ...)`
