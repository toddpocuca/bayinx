# Bayinx: <ins>Bay</ins>esian <ins>In</ins>ference with JA<ins>X</ins>

Bayinx is an embedded probabilistic programming language in Python, powered by [JAX](https://github.com/jax-ml/jax).
It is heavily inspired by and aims to have feature parity with [Stan](https://mc-stan.org/), but extends the types of objects you can work with and focuses on normalizing flows variational inference for sampling.

## Installation
Bayinx requires Python `3.12+`, JAX, and a few extra libraries in the JAX ecosystem.
The easiest way to get started is by installing from PyPi using your favourite python package manager. For example with `uv`:

```sh
# Ensure you are in your project environment
uv add bayinx
```

This installs the bare-bones version of Bayinx, however if you need additional functionality like GPU support, there are a couple of dependency groups:
```sh
# Ensure you are in your project environment
uv add 'bayinx[cuda]' # Installs Bayinx with CUDA support
```

## Documentation
Documentation is available at: https://toddpocuca.github.io/bayinx.

## Roadmap
- [ ] Allow shape definitions to include expressions (e.g., shape = 'n_obs + 1' will evaluate to the correct specification).
- [ ] Implement SALSA https://arxiv.org/abs/2002.10597.
- [ ] Nodes carry bounds for their support (i.e., node.obj ∈ [node._lb, node._ub]) which are used to check if inputs to distributions are valid (e.g., a node inputted as the scale of a normal dist must have `node._lb >= 0`).
