import jax.numpy as jnp
from jax.scipy.stats import norm

from bayinx.dists import Normal


def test_logprob():
    eps = jnp.finfo(jnp.array(0.0)).eps

    for mean, scale in [(0.0, 1.0), (3.14, 3.14), (2.31, 500)]:
        var = scale**2
        prec = 1 / var

        for x in [-390, 1, 30, 24]:
            assert abs(Normal(mean, scale).logprob(x) - norm.logpdf(x, mean, scale)) < eps
            assert abs(Normal(mean, var = var).logprob(x) - norm.logpdf(x, mean, scale)) < eps
            assert abs(Normal(mean, prec = prec).logprob(x) - norm.logpdf(x, mean, scale)) < eps
