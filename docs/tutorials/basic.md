# Basic Usage

If you are new to probabilistic programming or want to quickly get up to speed with Bayinx, this tutorial goes through how to specify and fit a model with Bayinx.

## Starting From A Statistical Model

Suppose we would like to specify a simple linear model, we would write this out like so:

$$
\begin{aligned}
    y_i &\sim \text{Normal}(\mu_i, \sigma) \\
    \mu_i &= \mathbf{x}_i^\top \mathbf{\beta} \\
\end{aligned}
$$

where $y_i \in \mathbb{R}$ denotes our response variable, $\mathbf{x}_i^\top \in \mathbb{R}^p$ denotes our predictors structured as a vector, and $\mathbf{\beta} \in \mathbb{R}^p$ denotes the parameters we wish to estimate.

or put in another way:

$$
    y_i = \mathbf{x}_i^\top \mathbf{\beta} + \epsilon_i
$$

where $\epsilon_i \sim \text{Normal}(0, \sigma)$.
But since these two formulations are equivalent, we will use the first.

## Translating to Bayinx

In Bayinx, model definitions are Python classes inheriting from `Model`, and just as we defined the model's logic and the objects used in said logic, we similarly have to declare both here:

```python
import jax.numpy as jnp
from jaxtyping import Array, Scalar

from bayinx import Model, Posterior, observed, stochastic
from bayinx.dists import Normal

# Define model
class LinearModel(Model):
    # The Objects Used By The Model
    beta: Array = stochastic(shape = 'n_pred')
    sigma: Scalar = stochastic(shape = (), lower = 0)

    X: Array = observed(shape = ('n_obs', 'n_pred'))
    y: Array = observed(shape = 'n_obs')

    # The Model's Logic
    def model(self, target):
        # Compute expected value
        mu = self.X @ self.beta

        # Define likelihood
        self.y << Normal(mu, self.sigma)
```

Notice the class's attributes are marked with field specifiers that denote whether an object is treated as a parameter (`stochastic`) or data (`observed`).
These specifiers do not *require* any arguments, but Bayinx offers some for user convenience:

### Shape Specification
The `shape` argument of `stochastic`/`observed` accepts a string, integer, or a tuple of strings and integers representing the shape of an array.
Depending on the type of the object it does two things:

- if the attribute is passed explicitly either through a [default initialization](../tutorials/basic.md#default-initialization) or during [posterior initialization](../tutorials/basic.md#fitting-a-model) (elaborated on below), the shape of the argument is checked against the shape specification. If the two disagree, an error will be thrown.
- if the attribute is `stochastic` and no object was passed explicitly, then an array is automatically constructed with the correct shape based on the shapes passed during posterior initialization.

This avoids boilerplate for many models that involve parameters defined as arrays, and peace of mind knowing that all arrays have the correct shape.

However, note that you do not NEED to specify the shape parameter!
It is always fine to leave the `shape` argument for an `observed` attribute blank, and as long as you specify the structure for a stochastic either with `init` (elaborated below) or during posterior initialization (elaborated further below), the same is true for a `stochastic` attribute.

### Default Initialization
The `init` argument of `stochastic`/`observed` is used to specify the default _structure_ of a stochastic attribute at "definition"-time (when you're writing your model), or the default _values_ for an observed attribute (note the distinction between _structure_ and _value_).
For example, if I would like a  to fallback to a list of arrays, that can be done like so:

```python
class ExampleModel(byx.Model):
    obj_1: list[Array] = stochastic(init = [jnp.ones(2), jnp.ones(3)])
    obj_2: list[Array] = observed(init = [jnp.ones(2), jnp.ones((2, 2))])

    # ...
```

Here, `init` specifies that `obj_1` looks like `[Array([#, #]), Array([#, #, #])]` since it is `stochastic` (note that the values given are _placeholders_, they are just used to get the correct structure).
For `obj_2` however, `init` specifies that it is exactly `[Array([1., 1.]), Array([1., 1., 1.])]` as it is `observed`!

These rules apply to initialization when constructing the posterior as well, for example, if we would like to override the default we can write:

```python
post = Posterior(
    ExampleModel,
    obj_1 = [jnp.ones(6), jnp.ones(7)] # override structure
)
```

### Constraints
Sometimes we would like to restrict a parameter to a certain subset of its domain, or ensure the data we've inputted satisfies certain conditions.
The last arguments of `stochastic`/`observed` are used to specify these constraints.
For example, a Bernoulli distribution typically involves a single parameter `p` that denotes the probabiliy of success, if we were to define a model for this, we could write:

```python
class BernoulliModel(byx.Model):
    p: Scalar = stochastic((), lower = 0, upper = 1)
    x: Array = observed()

    def model(self, target):
        self.x << byd.Bernoulli(self.p)
```

The bounds `lower` and `upper` are passed as arguments of `stochastic` and ensures that the posterior distribution for `p` is on the relevant domain.
Similarly we could add a constraint to `x`, as we know for a Bernoulli-distributed random variable it must lie between 0 and 1 (more specifically, it must be *either* 0 or 1).

## Fitting a Model

Once the model is defined we can proceed with fitting an approximation to the posterior distribution.
To initialize the posterior approximation, we pass all the necessary arguments to the `Posterior` class constructor including all observed data (if they do not have defaults), any shapes used, and lastly any `stochastic` attributes whose structure was not specified using `shape` or `init`.

Continuing with the `LinearModel` example from above, we construct `Posterior[LinearModel]` like so:

```python
# Initialize posterior approximation
post: Posterior[LinearModel] = Posterior(
    LinearModel,
    n_pred = 2,
    n_obs = 4,
    X = jnp.array([[1., 0.], [1., 1.], [1., 2.], [1., 3.]]),
    y = jnp.array([6.0, 13., 20., 27.])
)
```

Currently this approximation does not reflect the actual posterior (it is initialized with default parameters for the underlying variational distribution), but we can now configure and then fit it to get a useful variational approximation:

### Configuring and Fitting the Normalizing Flow Architecture

??? info "Confused: What is a normalizing flow?"
    Take a look at the overview on normalizing flows available [here](../nf.md).

Bayinx offers a variety of normalizing flow layers that can be used together in `bayinx.flows`, the simplest of which is the [diagonal affine layer](../api/flows.md#bayinx.flows.DiagAffine):

```py
from bayinx.flows import DiagAffine

post.configure([DiagAffine()])
```

We can then fit the approximation:

```py
post.fit()
```
```
Fitting Variational Approximation: 100%|████████████████████████████████████████████████████| 100000/100000 [00:00<00:00, 205572.70it/s]
```

## Generating Posterior Samples

Since we are given the variational approximation, new posterior samples can be generated as needed instead of needing to store a potentially large collection of draws:

```python
# Generate posterior draws
beta_draws = post.sample('beta', 10_000)

# Generate posterior predictive draws for a new datapoint
new_x = jnp.array([1., 4.])
ppred_draws = post.predictive(
    lambda model, key: Normal(new_x @ model.beta, model.sigma).sample(()),
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
