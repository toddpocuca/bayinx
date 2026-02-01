# Normalizing Flows Variational Inference

Variational inference (VI) methods have improved the scalability of Bayesian inference for practioners across domains, but the restricted family of approximations considered by current implementations limit their use-cases (e.g., ADVI can only approximate gaussian-like posteriors).
Normalizing flows (NFs) are a relatively recent effort to improve the ability of VI methods to approximate more complicated posterior densities.


## Normalizing Flows

Formally, a NF is defined as an indexed collection of invertible transformations--- called flows ---that map a base density $Z(\mathbf{\theta})$ to the
final variational distribution $Q_(\mathbf{\phi})(\mathbf{\theta})$.
In practice, a finite number of continuously parameterized flows $T_i : \mathbf{\Theta} \rightarrow \mathbf{\Theta}$ are used to construct the NF:

$$
\begin{aligned}
    \mathbf{\theta} &= T_n \circ \dots \circ T_1(\mathbf{\theta}_0) \\
    \ln Q_{\mathbf{\phi}}(\mathbf{\theta}) = \ln \tilde{Q}_{\mathbf{\phi}}(\mathbf{\theta}_0) &= \ln Z(\mathbf{\theta}_0) - \sum_{i=1}^n \ln \left| \det \frac{d}{d \mathbf{\theta}_{i-1}} [T_i](\mathbf{\theta}_{i-1}) \right|
\end{aligned}
$$

where since we have reparameterized a fixed base distribution $Q_0$ through a deterministic and differentiable mapping, the reparameterized gradient estimator can be used.

To keep this computationally efficient, Bayinx uses flows with triangular Jacobians.
This means the determinant used when evaluating the variational density can be simplified to the product of $D$ diagonal elements,
ensuring that even models with thousands of parameters remain fast.

## Choosing the Right Architecture

Since the default base distribution in Bayinx is a standard Student's T distribution (with learnable degrees of freedom), affine transformations will likely work for most models.
However, Bayinx also offers nonlinear flows to account for more complex posterior distributions.

### Affine Flows

* [Diagonal Affine](api/flows.md/#bayinx.flows.DiagAffine): Equivalent to [mean-field ADVI](https://mc-stan.org/docs/cmdstan-guide/variational_config.html) with a standard normal base distribution.
It scales and shifts each parameter independently, meaning it is cheap to compute but underestimates dispersion in the posterior when correlations are present.
* [Full Affine](api/flows.md/#bayinx.flows.FullAffine): Equivalent to [full-rank ADVI](https://mc-stan.org/docs/cmdstan-guide/variational_config.html) with a standard normal base distribution.
It improves the diagonal affine flow by using a lower-triangular matrix to capture every possible correlation between parameters, meaning it is highly expressive for Gaussian-like shapes, but memory-intensive.
* [Low-rank Affine](api/flows.md/#bayinx.flows.LowRankAffine): A middle ground that captures major correlations using a low-rank update, making it suitable for high-dimensional models where `FullAffine` would not fit in memory.

### Non-Linear Flows

If your posterior isn't just a "tilted ellipse" and has complex non-linear dependencies, stacking a series of [Sylvester flows]() will work for basically all cases.
