# Basic Usage

If you are new to probabilistic programming or want to quickly get up to speed with Bayinx, this tutorial goes through how to specify and fit a model with Bayinx.

## Starting From A Statistical Model

Suppose we would like to specify a simple linear model, in model notation this would be:

$$
    y_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \mathbf{x}_i^\top \mathbf{\beta}
$$
where $y_i \in \mathbb{R}$ denotes our response variable, $\mathbf{x}_i^\top \in \mathbb{R}^p$ denotes our predictors structured as a vector, and $\mathbf{\beta} \in \mathbb{R}^p$ denotes the true parameters we wish to estimate.

or put in another way:

$$
    y_i = \mathbf{x}_i^\top \mathbf{\beta} + \epsilon_i
$$
where $\epsilon_i \sim \text{Normal}(0, \sigma)$. But since these two formulations are equivalent, we will use the first.

## Translating to Bayinx

In Bayinx, model definitions are Python classes inheriting from `bayinx.Model`, and just as we defined the objects we used above in our model equations, we similarly have to declare the objects we wish to use in our model:

```python
import jax.numpy as jnp

import bayinx as byx
import bayinx.dists as byd
import bayinx.flows as byf
import bayinx.nodes as byn

# Define model
class LinearModel(byx.Model):
    beta: byn.Continuous = byx.define(shape = 'n_predictors')
    sigma: byn.Continuous = byx.define(shape = (), lower = 0)

    X: byn.Observed = byx.define(shape = ('n_obs', 'n_predictors'))
    y: byn.Observed = byx.define(shape = 'n_obs')

    def model(self, target):
        # Compute expected value
        mu = self.X @ self.beta

        # Define likelihood
        self.y << byd.Normal(mu, self.sigma)

# Initialize posterior approximation
post = byx.Posterior(
    LinearModel,
    n_predictors = 2,
    n_obs = 4,
    X = jnp.array([[1., 0.], [1., 1.], [1., 2.], [1., 3.]]),
    y = jnp.array([6.0, 13., 20., 27.])
)
```

Notice the attributes of the subclass are used to define _model nodes_, which are all the objects available to the model and represent data or parameters.
Model nodes are defined using `bayinx.define`, which is used to define shapes, default values, or constraints the node must satisfy:

### Shape Specification
The `shape` argument of `define` does two things:
- if the node is stochastic (e.g., `Continuous` inherits from `Stochastic`, so it is stochastic and represents a parameter), then an (JAX) array is automatically constructed with the correct shape based on the shapes passed during posterior initialization (elaborated on later).
- if the node is passed explicitly during posterior initialization, the shape of the argument is checked against the shape specification. If the two disagree, an error will be thrown.

This avoids some boilerplate for many models that involve parameters defined as arrays, and peace of mind knowing that all model nodes have the correct shape.

However, note that you do not NEED to specify the shape parameter!
It is always fine to leave the `shape` argument for an `Observed` node blank, and as long as you specify the structure for a stochastic either with `init` or during posterior initialization, the same is true for any `Stochastic` node.

### Default Initialization
The `init` argument of `define` is used to specify the default structure of a stochastic node at "definition"-time (when you're writing your model), or the default values for an observed node.
For example, if I would like a model node to fallback to a list of arrays, that can be done like so:

```python
import jax.numpy as jnp

class ExampleModel(byx.Model):
    node_1: byn.Continuous = byx.define(init = [jnp.ones(2), jnp.ones(3)])
    node_2: byn.Observed = byx.define(init = [jnp.ones(2), jnp.ones((2, 2))])

    # ...
```

Here, `init` specifies that `node_1` looks like `[Array([#, #]), Array([#, #, #])]` since it is `Continuous` (note that the values given are _placeholders_, they are just used to get the correct structure).
For `node_2` however, `init` specifies that it is exactly `[Array([1., 1.]), Array([1., 1., 1.])]` as it is `Observed`!

These rules apply to initialization when constructing the posterior as well, for example, if we would like to override the default we can write:

```python
post = byx.Posterior(
    ExampleModel,
    node_1 = [jnp.ones(6), jnp.ones(7)] # new structure
)
```

### Constraints
Sometimes we would like to restrict a parameter to a certain subset of its domain, or ensure the data we've inputted satisfies certain conditions.
The last arguments of `define` are used to specify these constraints.
For example, a Bernoulli distribution involves a single parameter `p` that denotes the probabiliy of success, if we were to define a model for this, we could write:

```python
class BernoulliModel(byx.Model):
    p: Continuous = byx.define((), lower = 0, upper = 1)
    x: Observed = byx.define()

    def model(self, target):
        self.x << byd.Bernoulli(self.p)

post = byx.Posterior(
    x = jnp.array([0, 0, 0, 1])
)
```


## Fitting a Model

Once the model is defined and the posterior is initialized, Bayinx uses Normalizing Flows Variational Inference to optimize a variational approximation.
The architecture is specified using the `.configure` method of `Posterior` using the flows offered in the `bayinx.flows` module.
The approximation can then optimized using the `.fit` method.
Continuing with the `LinearModel` example, a full affine flow can be used to accurately approximate the posterior:

```python
# Configure and optimize posterior
post.configure([byf.FullAffine()]) # equivalent to full-rank ADVI
post.fit()
```
```
Fitting Variational Approximation: 100%|███████████████| 100000/100000 [00:01<00:00, 55788.50it/s]
```

## Generating Posterior Samples

Since we are given the variational approximation, new posterior samples can be generated as needed instead of needing to store a potentially large collection of draws:

```python
# Generate posterior draws
beta_draws = post.sample('beta', 10_000)

# Generate posterior predictive draws for a new datapoint
new_x = jnp.array([1., 4.])
ppred_draws = post.predictive(
    lambda model, key: byd.Normal(new_x @ model.beta, model.sigma).sample(()),
    10_000
)

# Print posterior means
print(f"Posterior mean of 'beta': {beta_draws.mean(0)}")
print(f"Posterior predictive mean of new datapoint: {ppred_draws.mean(0)}")
```
```
Posterior mean of 'beta': [6.0002875 6.9998145]
Posterior predictive mean of new datapoint: 34.00600814819336
```
