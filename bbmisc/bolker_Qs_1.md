## Sample workflow

Say I have data and a model in mind. I'd love an example showing how I run everything (e.g. take an example from `statsmodels` and run it/compare ...)

Is beta automatically vectorized? That is, if I have a Poisson with multiple predictors, are the parameter priors iid Normal(0,10)? (I think so.) Is there a way to avoid hard-coding the prior parameters?  Should I define a function with prior params as arguments that returns a model object?

Which library defines @ as matrix multiplication?

I'm still trying to figure out how to run stuff interactively. I tried 

```
uv run --with bayinx_example/ jupyter-notebook
```

https://docs.astral.sh/uv/guides/integration/jupyter/

after setting up the project,

```
uv run --with jupyter jupyter notebook
```

works

```
uv add ArViZ
```

Would be nice if arviz knew about `jax.ArrayImpl` objects ...

> ValueError: Can only convert xarray dataarray, xarray dataset, dict, pytree (if 'dm-tree' is installed), netcdf filename, numpy array, pystan fit, emcee fit, pyro mcmc fit, numpyro mcmc fit, cmdstan fit csv filename, cmdstanpy fit to InferenceData, not ArrayImpl

hmm, maybe this is already handled in a development version of arviz?  https://github.com/arviz-devs/arviz/issues/2480 was closed/implemented on October 2 ... (don't easily know how to check Python package versions ...)
