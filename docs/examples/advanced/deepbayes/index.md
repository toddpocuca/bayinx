# Deep Bayesian Neural Networks

## What is a Bayesian Neural Network?

A standard neural network is trained to find a single set of weights $\mathbf{w}$ that minimises some loss.
This point estimate says nothing about how *certain* the network is in its predictions — a network can be confidently wrong, especially far from the training data.

A **Bayesian Neural Network (BNN)** instead treats the weights as random variables and places a prior distribution over them:

$$p(\mathbf{w}) = \prod_i \mathcal{N}(w_i \mid 0, 1)$$

After observing data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}$, we seek the posterior:

$$p(\mathbf{w} \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})$$

Predictions are then made by integrating over all plausible weight configurations — a process called **posterior predictive inference**:

$$p(y^* \mid \mathbf{x}^*, \mathcal{D}) = \int p(y^* \mid \mathbf{x}^*, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, d\mathbf{w}$$

This gives us **uncertainty-aware predictions**: the model can express that it doesn't know the answer, not just what its best guess is.

## Why "Deep"?

A *deep* BNN simply uses a multi-layer (deep) neural network within the model logic, for example parameterizing the functional form of the effect of some predictors on a response variable.
Each layer applies a linear transformation followed by a nonlinearity, building up increasingly abstract representations of the input.
The posterior is then placed over the weights of all layers simultaneously. Depth brings expressive power; the Bayesian treatment brings calibrated uncertainty.

## Inference via Variational Inference

The exact posterior $p(\mathbf{w} \mid \mathcal{D})$ is intractable for any non-trivial network. Bayinx approximates it using **variational inference with normalizing flows**:
we find a flexible distribution $q_\phi(\mathbf{w})$ that minimises the KL divergence to the true posterior by maximising the Evidence Lower Bound (ELBO).

---

## Example: Fitting a BNN with Bayinx and Equinox

Below is a complete example fitting a deep BNN to a 1-D regression problem.
The network is defined with [Equinox](https://docs.kidger.site/equinox/) and embedded inside a Bayinx `Model` as a `stochastic` node, so every weight and bias is automatically treated as a latent variable.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import bayinx as byx
import bayinx.dists as byd
import bayinx.flows as byf
from jaxtyping import Array, PRNGKeyArray, Scalar


class MyNeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, n_in: int, key: PRNGKeyArray):
        k1, k2, k3 = jr.split(key, 3)
        self.layers = [
            eqx.nn.Linear(n_in, 5, key=k1),
            eqx.nn.Linear(5, 5, key=k1),
            eqx.nn.Linear(5,   1, key=k2),
        ]

    def __call__(self, x: Array) -> Array:
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x).squeeze()


class MyModel(byx.Model):
    nn: MyNeuralNetwork = byx.stochastic(
        init=MyNeuralNetwork(n_in=1, key=jr.key(0))
    )
    sigma: Scalar = byx.stochastic(shape=(), lower=0.0)

    X: Array = byx.observed(shape=("n_obs", "n_in"))
    y: Array = byx.observed(shape="n_obs")

    def model(self, target):
        # Priors
        self.nn << byd.Normal(0.0, 1.0) # constrain weights
        self.sigma << byd.Exponential(1.0)

        # Likelihood
        mu = jax.vmap(self.nn)(self.X)
        self.y << byd.Normal(mu, self.sigma)


n_obs, n_in = 1000, 1
X_data = jr.uniform(jr.key(0), (n_obs, n_in), minval=-3.0, maxval=3.0)
y_data = jnp.sin(X_data[:, 0]) + 0.2 * jr.normal(jr.key(1), (n_obs,))

posterior = byx.Posterior(
    MyModel,
    n_obs=n_obs, n_in=n_in,
    X=X_data, y=y_data,
    nn=MyNeuralNetwork(n_in=n_in, key=jr.key(99)),
)

posterior.configure([byf.DiagAffine()])
posterior.fit(max_iters=1_000_000, stl=True)

X_test = jnp.array([-jnp.pi / 2, 0, jnp.pi/2])[:, None]

def predict(model: MyModel, key: jax.Array) -> Array:
    mu = jax.vmap(model.nn)(X_test)
    return byd.Normal(mu, model.sigma).sample(mu.shape, key=key)

ppred = posterior.predictive(predict, n_draws=1_000)  # shape (1_000, 100)

print("Predictive mean:", ppred.mean(0))
print("Predictive std: ", ppred.std(0))
```
```
Predictive mean: [-0.96822405 -0.00274488  0.9682781 ]
Predictive std:  [0.20083815 0.20931077 0.20635626]
```
