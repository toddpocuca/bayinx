# Coming From Stan to Bayinx

I have been an avid Stan user for a few years now and got inspired to write my own probabilistic programming language, Bayinx.
If you are experienced than Stan then this will be a useful tutorial for understanding Bayinx.

## Defining Models In Stan and Bayinx

To highlight the similarities between Stan and Bayinx, we'll look at a simple example where the data $X_i$ are normally distributed with an unknown mean $\mu$ and standard deviation $\sigma$:

$$
\begin{aligned}
    X_i &\sim \text{Normal}(\mu, \sigma) \\
    \mu &\sim \text{Normal}(0, 10) \\
    \sigma &\sim \text{Exponential}(10)
\end{aligned}
$$

### Stan Implementation
In Stan, you would write the above model as:

```stan
data {
    int<lower=1> n_obs;
    vector[n_obs] x;
}
parameters {
    real mu;
    real<lower=0> sigma;
}
model {
    // Defining priors
    mu ~ normal(0, 10);
    sigma ~ exponential(10);

    // Defining likelihood
    sigma ~ exponential(10);
}
```

We would then either use `cmdstan` or our favourite library (`CmdStanR`, `CmdStanPy`, etc) to pass in our data and fit the model with Stan.

### Bayinx Implementation
In Bayinx, we create a new class that inherits from `bayinx.Model`, and use attribute annotations & the `define` function to construct the nodes for our model.

```py
from bayinx.dists import Normal, Exponential
from bayinx.nodes import Continuous, Observed
from bayinx import Model, define

class SimpleNormalModel(Model):
    mu: Continuous = define(shape=())
    sigma: Continuous = define(shape=(), lower=0)

    x: Observed = define(shape='n_obs')

    def model(self, target):
        # Defining priors
        self.mu << Normal(0, 10)
        self.sigma << Exponential(scale = 10)

        # Defining likelihood
        self.x << Normal(self.mu, self.sigma)

        return target
```

The `data` and `parameters` blocks are essentially merged into one "block"; defining an object as data or parameter is done by defining a new attribute.

The `model` block in Stan is equivalent to the `model` method of our new class, which takes in `self` and the accumulator for the log-probability evaluations `target`, and returns it.
Just like in Stan, you often do not need to work with `target` explicitly, instead using distribution statements with `<<` (as opposed to `~` in Stan).

Similar to Stan, distribution statements are broadcasted (even across [PyTrees](https://docs.jax.dev/en/latest/pytrees.html)!) & vectorized.

Fitting the model is done by passing the model definition and shapes/data to `Posterior`, constructing a specification for the normalizing flow architecture, and the optimizing the variational approximation:
```py
from bayinx import Posterior
from bayinx.flows import DiagAffine
import jax.numpy as jnp

post = Posterior(
    SimpleNormalModel,
    n_obs = 3,
    x = jnp.array([-1.0, 0.0, 1.0])
)
post.configure([DiagAffine()])
post.fit()
```

This highlights most of the important similarities between Stan and Bayinx, but there are some important differences the two.

## Differences Between Bayinx & Stan

### Who Needs Shapes!
You don't technically *need* to define the shape of a node in Bayinx, it's offered so we can perform shape-checks during initialization, but recall above how we used `n_obs` to define the shape of `x`? We could've just written:

```py
class SimpleNormalModel(Model):
    mu: Continuous = define(shape=())
    sigma: Continuous = define(shape=(), lower=0)

    x: Observed = define()

    def model(self, target):
        # Defining priors
        self.mu << Normal(0, 10)
        self.sigma << Exponential(scale = 10)

        # Defining likelihood
        self.x << Normal(self.mu, self.sigma)

        return target

post = Posterior(
    SimpleNormalModel,
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]) # This can hold any shape!
)
post.configure([DiagAffine()])
post.fit()
```


In fact, we can even drop the shape definitions on *everything*:

```py
class SimpleNormalModel(Model):
    mu: Continuous = define()
    sigma: Continuous = define(lower=0)

    x: Observed = define()

    def model(self, target):
        # Defining priors
        self.mu << Normal(0, 10)
        self.sigma << Exponential(scale = 10)

        # Defining likelihood
        self.x << Normal(self.mu, self.sigma)

        return target

post = Posterior(
    SimpleNormalModel,
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
    mu = jnp.zeros(()),
    sigma = jnp.zeros(())
)
post.configure([DiagAffine()])
post.fit()
```

When we pass (named) arguments to `Posterior`, it uses them to initialize a "toy" model with the correct structure for the variational approximation (so we can figure out the size of the parameter space).
These arguments are the shapes for the nodes (so we can perform shape-checks on inputs and/or automatically generate parameter nodes with the correct structure for you), and in some sense the actual nodes themselves: we can pass a toy array with the shape we want and internally this will be used to generate a node with the same shape.


### Nodes Can Be *Anything*

Well not exactly anything, but they can be a lot more than just arrays.

For example, we can work with objects as simple as a list of arrays:

```py
class MyModel(Model):
    mu: Continuous = define()
    sigma: Continuous = define(lower=0)

    x: Observed = define(shape = 'x_shape')

    def model(self, target):
        # Defining priors
        self.mu << Normal(0, 10)
        self.sigma << Exponential(0.1) # Defaults to rate parameterization

        # Defining likelihood
        for x, mu, sigma in zip(self.x, self.mu, self.sigma):
            x << Normal(mu, sigma)

        return target

post = Posterior(
    MyModel,
    x_shape = (2, 3),
    x = jnp.array([[-11.0, -10.0, -9.0], [9.0, 10.0, 11.0]]),
    mu = [0.0, 0.0],
    sigma = [0.0, 0.0]
)
post.configure([DiagAffine()])
post.fit()
```

To something as complicated as a neural network:

```py
from functools import partial
import equinox as eqx
import jax
import jax.random as jr

# Define neural network
class MyNeuralNetwork(eqx.Module):
    layers: list

    def __init__(self):
        self.layers = [
            eqx.nn.Linear('scalar', 20, key=jr.key(0)),
            jax.nn.leaky_relu,
            eqx.nn.Linear(20, 'scalar', key=jr.key(1))
        ]

    @partial(eqx.filter_vmap, in_axes = (None, 0))
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Define model
class NeuralNetworkModel(Model):
    nn: Continuous = define(
        init = MyNeuralNetwork() # if a node is known at "definition"-time we can pass it here
    )
    sigma: Continuous = define(shape = (), lower = 0.0)

    x: Observed = define(shape = 'n_obs')
    y: Observed = define(shape = 'n_obs')

    def model(self, target):
        # Set prior to constrain weights
        self.nn << Normal(0, 3)

        # Compute expected response
        mu = self.nn(self.x)

        # Accumulate likelihood
        self.y << Normal(mu, self.sigma)

        return target

# Approximate a sine function
n_obs = 1000
x = jr.uniform(jr.key(0), (n_obs, ), minval = -jnp.pi, maxval = jnp.pi)
y = jnp.sin(x)

# Construct posterior
post = Posterior(
    NeuralNetworkModel,
    n_obs = n_obs,
    x = x,
    y = y
)
post.configure([DiagAffine()])
post.fit(int(1e5), grad_draws = 2, batch_size = 2)

# Get predictives on new data
x_new = jnp.array([-jnp.pi, -jnp.pi/2, 0, jnp.pi / 2, jnp.pi])
y_new = jnp.sin(x_new)
y_newhat = post.predictive(lambda model, key: model.nn(x_new), 1000)

print(f"Ground-truth For New Data: {y_new}")
print(f"Posterior Predictive Mean For New Data: {y_newhat.mean(0)}")
print(f"Difference: {y_new - y_newhat.mean(0)}")
```
```bash
Ground-truth For New Data: [ 8.742278e-08 -1.000000e+00  0.000000e+00  1.000000e+00 -8.742278e-08]
Posterior Predictive Mean For New Data: [-0.03101175 -0.98273844 -0.00559028  0.9660495   0.06396807]
Difference: [ 0.03101183 -0.01726156  0.00559028  0.03395051 -0.06396816]
```
That's pretty good for a small network trained with only 100 000 iterations.

### Parallelization Out of the Box

One of my issues with Stan is that there is some rewriting involved in trying to get multi-threading to work, and some work optimizing the grainsize (although a maximal grainsize `n_elements // n_threads` seems to work best in my experience). Thankfully, XLA automatically scales the number of threads used with the size of the problem via [cost modelling](https://github.com/openxla/xla/blob/ba2ef9892875a41eb9f30efb2582d8728dc6b9d8/xla/service/cpu/parallel_task_assignment.cc#L81), and both pre-allocates the memory used for an entire program as well as aggressively optimizing memory usage to avoid unnecessary copies (like [duplicating arguments shared amongst threads](https://discourse.mc-stan.org/t/reduce-sum-results-in-much-slower-run-times-even-for-large-datasets/26827/3)). Meaning you don't have to modify your Bayinx model at all to take advantage of multi-threading.
