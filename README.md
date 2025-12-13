# Bayinx: <ins>Bay</ins>esian <ins>In</ins>ference with JA<ins>X</ins>

Bayinx is an embedded probabilistic programming language in Python, powered by
[JAX](https://mc-stan.org/). It is heavily inspired by and aims to have
feature parity with [Stan](https://mc-stan.org/), but extends the types of
objects you can work with and focuses on normalizing flows variational
inference for sampling.

## Roadmap
- [ ] Implement OT-Flow: https://arxiv.org/abs/2006.00104
- [ ] Allow shape definitions to include expressions (e.g., shape = 'n_obs + 1' will evaluate to the correct specification)
- [ ] Find a nice way to track the ELBO trajectory to implement early stoppage (tolerance currently does nothing).
- [ ] Allow users to specify custom tolerance criteria.
