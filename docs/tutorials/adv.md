# Advanced Usage

This tutorial covers Bayinx's loop utilities — `map` and `fori_loop` — which let you process data element-by-element inside a model while still accumulating the log-density correctly and remaining fully compatible with JAX compilation.

In many models you naturally want to iterate over observations — computing per-observation quantities, applying different transformations, or building up contributions to the likelihood one at a time.
Python's built-in `for` loops work fine in Bayinx in general, but when those loops need to run inside a `jax.jit`-compiled function (which all fitted models do), the loop must be *static*, meaning the number of iterations is fixed at compile time and JAX unrolls every iteration into the computation graph.
For small loops this is fine, but for large datasets it produces enormous, slow-to-compile graphs.

JAX offers a variety of functions to keep the computation graph small by compiling the inner work of the loop once and iterating with `map`, `scan`, `fori_loop`, `while_loop`, etc.
However we cannot use distribution statements natively with these functions, and so Bayinx offers alternatives with the same functionality.
In other words, they are context-aware: when called inside a model's `model` method, they automatically accumulate each iteration's log-probability contribution into the running target density, so you never have to manage that bookkeeping yourself.

## Importing the Utilities

```python
from bayinx.ops import map, fori_loop
```

## `map` — Mapping Over Data

`map` applies a function over the *leading axis* of one or more arrays, accumulating log-probability contributions along the way.

```python
map(f, *xs)
```

- `f` is a callable that receives a single element (or a tuple of elements, one from each array in `xs`) and optionally returns a value.
- `xs` are the arrays to map over. They are sliced along their leading axis and passed positionally to `f`.
- If `f` returns values, `map` stacks them and returns the result; if `f` returns nothing, `map` returns `None`.

## `fori_loop` — Looping Over Indices

`fori_loop` is the index-based counterpart: it iterates an integer index from `lower` to `upper` (exclusive), calling `f(i)` at each step.

```python
fori_loop(lower, upper, f)
```

- `f` receives the current integer index `i`.
- Like `map`, it accumulates log-probability contributions inside a model context and stacks any returned values.

## Example: A GLM With Per-Observation Likelihood

To show both utilities in a realistic setting, consider a Poisson GLM where we process each observation individually.
The model is:

$$
\begin{aligned}
    y_i &\sim \text{Poisson}(\lambda_i) \\
    \log(\lambda_i) &= \mathbf{x}_i^\top \boldsymbol{\beta} \\
    \boldsymbol{\beta} &\sim \text{Normal}(\mathbf{0},\, 3)
\end{aligned}
$$

### Setup

```python
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from bayinx import Model, Posterior, stochastic, observed
from bayinx.dists import Normal, Poisson
from bayinx.ops import map, fori_loop

# Simulate some data
key = jr.key(0)
n_obs, n_pred = 200, 3
X = jr.normal(key, (n_obs, n_pred))
true_beta = jnp.array([0.5, -1.0, 0.3])
lam_true = jnp.exp(X @ true_beta)
y = jr.poisson(jr.key(1), lam_true)
```

### Using `map`

`map` is the most natural fit here: we have a fixed array of observations and want to apply the same likelihood function to each row.

```python
class PoissonGLM_Map(Model):
    beta: Array = stochastic(shape='n_pred')

    X: Array = observed(shape=('n_obs', 'n_pred'))
    y: Array = observed(shape='n_obs')

    def model(self, target):
        # Prior on coefficients
        self.beta << Normal(0, 3)

        # Process each observation separately using map
        def observation_likelihood(xi, yi):
            lam = jnp.exp(xi @ self.beta)
            yi << Poisson(lam)

        map(observation_likelihood, self.X, self.y)

post = Posterior(
    PoissonGLM_Map,
    n_obs=n_obs,
    n_pred=n_pred,
    X=X,
    y=y
)
```

`map` slices `self.X` and `self.y` along their leading axis, passing one row `xi` and one scalar `yi` to `observation_likelihood` per iteration.
Each call to `yi << Poisson(lam_i)` contributes to the log-density, and `map` accumulates all those contributions back into the outer model context.

### Using `fori_loop`

`fori_loop` is useful when you need explicit index access — for example, to index into arrays differently for each step, or when the loop bounds come from a shape rather than an array.

```python
class PoissonGLM_Fori(Model):
    beta: Array = stochastic(shape='n_pred')

    X: Array = observed(shape=('n_obs', 'n_pred'))
    y: Array = observed(shape='n_obs')

    def model(self, target):
        # Prior on coefficients
        self.beta << Normal(0, 3)

        # Process each observation separately using fori_loop
        def observation_likelihood(i):
            lam = jnp.exp(self.X[i] @ self.beta)
            self.y[i] << Poisson(lam)

        fori_loop(0, self.y.shape[0], observation_likelihood)

post = Posterior(
    PoissonGLM_Fori,
    n_obs=n_obs,
    n_pred=n_pred,
    X=X,
    y=y
)
```

The two approaches produce identical results — choose whichever reads more clearly for your model.

### Fitting and Sampling

Both model variants are fitted the same way as any other Bayinx model:

```python
from bayinx.flows import DiagAffine

post.configure([DiagAffine()])
post.fit()

beta_draws = post.sample('beta', 10_000)
print(f"True beta:            {true_beta}")
print(f"Posterior mean (beta): {beta_draws.mean(0)}")
```
```
Fitting Variational Approximation: 100%|████████████████████████████████████████| 100000/100000 [00:01<00:00, 67082.35it/s]
True beta:            [ 0.5 -1.   0.3]
Posterior mean (beta): [ 0.53247786 -0.91966873  0.38757807]
```

## Collecting Return Values

Both `map` and `fori_loop` can also collect values returned by `f`, which is useful for computing per-observation quantities you want to inspect after fitting.

```python
def compute_lambda(model, key):
    def per_obs(xi):
        return jnp.exp(xi @ model.beta)

    return map(per_obs, model.X)  # returns shape (n_obs,) for each draw

lambda_draws = post.predictive(compute_lambda, 1000)
print(f"True lambda[0]:            {lam_true[0]}")
print(f"Posterior mean of lambda[0]: {lambda_draws.mean(0)[0]}")
```
```
True lambda[0]:            0.26079463958740234
Posterior mean of lambda[0]: 0.3157551884651184
```

When `f` returns a value, the results are stacked along a new leading axis, giving you an `(n_draws, ...)` array. When `f` returns nothing (as in the likelihood-only examples above), both utilities return `None`.
