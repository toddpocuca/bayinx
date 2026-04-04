from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer, FlowSpec


def gather(arr, i) -> Array:
    return jnp.take_along_axis(arr, i[:, None], axis=1).squeeze()

def lr_forward(
    x: Array,
    xk_lhs: Array,
    xk_rhs: Array,
    yk_lhs: Array,
    yk_rhs: Array,
    d_lhs: Array,
    d_rhs: Array,
    lam: Array
) -> tuple[Array, Array]:
    """
    Compute the output of a linear rational function & its elementwise log-derivative.
    """
    eps = 0.0

    # Compute width, height & total slope for bin
    width = jnp.maximum(xk_rhs - xk_lhs, eps)
    height = jnp.maximum(yk_rhs - yk_lhs, eps)
    avg_slope = height / width

    # Compute relative position within bin
    phi = (x - xk_lhs) / width

    # Compute rhs weight
    w_rhs = jnp.sqrt(d_lhs / d_rhs) # w_lhs is implicitly 1

    # Compute output at midpoint
    y_mid_num = (1 - lam) * yk_lhs + lam * w_rhs * yk_rhs
    y_mid_den = (1 - lam) + lam * w_rhs
    y_mid = y_mid_num / y_mid_den

    # Compute midpoint weight
    w_mid = (lam * d_lhs + (1 - lam) * w_rhs * d_rhs) / avg_slope

    # Compute output & its derivative in left segment
    y_lhs_num = yk_lhs * (lam - phi) + w_mid * y_mid * phi
    y_lhs_den = (lam - phi) + w_mid * phi
    y_in_lhs = y_lhs_num / y_lhs_den
    deriv_in_lhs = lam * w_mid * (y_mid - yk_lhs) / y_lhs_den**2

    # Compute output & its derivative in right segment
    y_rhs_num = w_mid * y_mid * (1 - phi) + w_rhs * yk_rhs * (phi - lam)
    y_rhs_den = w_mid * (1 - phi) + w_rhs * (phi - lam)
    y_in_rhs = y_rhs_num / y_rhs_den
    deriv_in_rhs = (1 - lam) * w_mid * w_rhs * (yk_rhs - y_mid) / y_rhs_den**2

    # Select the segment
    y = jnp.where(phi < lam, y_in_lhs, y_in_rhs)
    deriv = jnp.where(phi < lam, deriv_in_lhs, deriv_in_rhs)

    return y, jnp.log(deriv) - jnp.log(width)


def lr_reverse(
    y: Array,
    xk_lhs: Array,
    xk_rhs: Array,
    yk_lhs: Array,
    yk_rhs: Array,
    d_lhs: Array,
    d_rhs: Array,
    lam: Array
) -> tuple[Array, Array]:
    """
    Compute the inverse of a linear rational function & its elementwise log-derivative.
    """
    eps = 0.0

    # Compute width, height & total slope for bin
    width = jnp.maximum(xk_rhs - xk_lhs, eps)
    height = jnp.maximum(yk_rhs - yk_lhs, eps)
    avg_slope = height / width

    # Compute rhs weight
    w_rhs = jnp.sqrt(d_lhs / d_rhs) # w_lhs is implicitly 1

    # Compute output at midpoint
    y_mid_num = (1 - lam) * yk_lhs + lam * w_rhs * yk_rhs
    y_mid_den = (1 - lam) + lam * w_rhs
    y_mid = y_mid_num / y_mid_den

    # Compute midpoint weight
    w_mid = (lam * d_lhs + (1 - lam) * w_rhs * d_rhs) / avg_slope

    # Compute x & its derivative in left segment
    phi_lhs_num = lam * (yk_lhs - y)
    phi_lhs_den = (yk_lhs - y) + w_mid * (y - y_mid)
    phi_in_lhs = phi_lhs_num / phi_lhs_den
    deriv_in_lhs = lam * w_mid * (y_mid - yk_lhs) / phi_lhs_den**2

    # Compute x & its derivative in right segment
    phi_rhs_num = lam * w_rhs * (yk_rhs - y) + w_mid * (y - y_mid)
    phi_rhs_den = w_rhs * (yk_rhs - y) + w_mid * (y - y_mid)
    phi_in_rhs = phi_rhs_num / phi_rhs_den
    deriv_in_rhs = (1 - lam) * w_mid * w_rhs * (yk_rhs - y_mid) / phi_rhs_den**2

    # Select the segment
    phi = jnp.where(y < y_mid, phi_in_lhs, phi_in_rhs)
    deriv = jnp.where(y < y_mid, deriv_in_lhs, deriv_in_rhs)

    # Compute absolute value from relative position
    x = phi * width + xk_lhs

    return x, jnp.log(deriv) + jnp.log(width)

class LinearRationalSplineLayer(FlowLayer):
    """
    A linear-rational spline flow layer.

    # Attributes
    - `params`: The parameters of the flow layer.
    - `constraints`: The constraining transformations for the parameters flow.
    - `static`: Whether the flow layer is frozen (parameters are not subject to further optimization).
    - `n_bins`: The number of bins of the spline.
    """
    params: Dict[str, Array]
    constraints: Dict[str, Callable[[Array], Array]]
    static: bool
    n_bins: int
    dim: int

    def __init__(self, dim: int, n_bins: int, key: PRNGKeyArray):
        """
        Initializes a linear rational spline flow layer.

        # Parameters
        - `dim`: The dimension of the parameter space.
        - `n_bins`: The number of bins.
        - `key`: The PRNG key for initializing parameters.
        """
        self.static = False
        self.n_bins = n_bins
        self.dim = dim

        # Split key
        k1, k2, k3, k4, k5, k6, k7, k8 = jr.split(key, 8)

        self.params = {
            "input_knots": jr.normal(k1, (dim, n_bins)) * 0.1 / (dim * n_bins)**0.5,
            "output_knots": jr.normal(k2, (dim, n_bins)) * 0.1 / (dim * n_bins)**0.5,
            "lam": jr.normal(k3, (dim, n_bins)) * 0.1 / (dim * n_bins)**0.5,
            "derivatives": jr.normal(k4, (dim, n_bins - 1)) * 0.1 / (dim * n_bins)**0.5,
            "scale": jr.normal(k6, (dim, )) * 0.1 / dim**0.5,
            "shift": jr.normal(k8, (dim, )) * 0.1 / dim**0.5
        }

        # Define constraints
        self.constraints = {
            "input_knots": lambda x: jnn.softmax(x, axis = 1),
            "output_knots": lambda x: jnn.softmax(x, axis = 1),
            "lam": jnn.sigmoid,
            "derivatives": jnp.exp,
            "scale": jnp.exp
        }

    def transform_params(self):
        params = self.constrain_params().copy()

        # Extract boundary information
        scale: Float[Array, "dim 1"] = params["scale"][:, None]
        shift: Float[Array, "dim 1"] = params["shift"][:, None]

        # Transform unconstrained knots
        widths = jnp.cumsum(params["input_knots"] * (2 * scale), axis=1)
        heights = jnp.cumsum(params["output_knots"] * (2 * scale), axis=1)

        params["input_knots"] = jnp.concat(
            [jnp.zeros((self.dim, 1)), widths],
            axis = 1
        ) + shift - scale

        params["output_knots"] = jnp.concat(
            [jnp.zeros((self.dim, 1)), heights],
            axis = 1
        ) + shift - scale

        # Pad boundary derivatives with 1.0 for identity transform
        ones = jnp.ones((self.dim, 1))
        params["derivatives"] = jnp.concat(
            [ones, params["derivatives"], ones], axis=1
        )

        return params


    def __forward(self, draw: Float[Array, " dim"], params: dict[str, Array]) -> Float[Array, " dim"]:
        assert len(draw.shape) == 1

        # Extract parameters
        input_knots: Float[Array, "dim n_bins+1"] = params["input_knots"]
        output_knots: Float[Array, "dim n_bins+1"] = params["output_knots"]
        lam: Float[Array, "dim n_bins"] = params["lam"]
        derivatives: Float[Array, "dim n_bins+1"] = params["derivatives"]

        # Match each element to its respective bin
        idx = jax.vmap(lambda x, k: jnp.searchsorted(k, x) - 1)(draw, input_knots)
        idx = jnp.clip(idx, 0, self.n_bins - 1)

        # Left of draw
        xk_lhs = gather(input_knots, idx)
        yk_lhs = gather(output_knots, idx)
        d_lhs  = gather(derivatives, idx)

        # Right of draw
        xk_rhs = gather(input_knots, idx + 1)
        yk_rhs = gather(output_knots, idx + 1)
        d_rhs  = gather(derivatives, idx + 1)

        # Relative midpoint for selected bin
        lam = gather(lam, idx)

        # Compute forward transformation ----
        output_spline, _ = lr_forward(
            draw,
            xk_lhs,
            xk_rhs,
            yk_lhs,
            yk_rhs,
            d_lhs,
            d_rhs,
            lam
        )

        # Grab bounds
        llhs_ik = input_knots[:, 0]
        rrhs_ik = input_knots[:, -1]

        # Use identity transform outside boundary
        draw = jnp.where(draw < llhs_ik, draw, jnp.where(draw > rrhs_ik, draw, output_spline))

        return draw

    @eqx.filter_jit
    def forward(self, draws: Float[Array, "draws dim"]) -> Float[Array, "draws dim"]:
        f = jax.vmap(self.__forward, (0, None))
        return f(draws, self.transform_params())

    def __reverse(self, draw: Float[Array, " dim"], params: dict[str, Array]) -> Float[Array, " dim"]:
        assert len(draw.shape) == 1

        # Extract parameters
        input_knots: Float[Array, "dim n_bins+1"] = params["input_knots"]
        output_knots: Float[Array, "dim n_bins+1"] = params["output_knots"]
        lam: Float[Array, "dim n_bins"] = params["lam"]
        derivatives: Float[Array, "dim n_bins+1"] = params["derivatives"]

        # Match each element to its respective bin
        idx = jax.vmap(lambda y, k: jnp.searchsorted(k, y) - 1)(draw, output_knots)
        idx = jnp.clip(idx, 0, self.n_bins - 1)

        # Left of draw
        xk_lhs = gather(input_knots, idx)
        yk_lhs = gather(output_knots, idx)
        d_lhs  = gather(derivatives, idx)

        # Right of draw
        xk_rhs = gather(input_knots, idx + 1)
        yk_rhs = gather(output_knots, idx + 1)
        d_rhs  = gather(derivatives, idx + 1)

        # Relative midpoint for selected bin
        lam = gather(lam, idx)

        # Compute reverse transformation
        input_spline, _ = lr_reverse(
            draw, xk_lhs, xk_rhs, yk_lhs, yk_rhs, d_lhs, d_rhs, lam
        )

        # Use identity transform outside boundary defined by output knots
        llhs_ok = output_knots[:, 0]
        rrhs_ok = output_knots[:, -1]
        draw = jnp.where(draw < llhs_ok, draw, jnp.where(draw > rrhs_ok, draw, input_spline))

        return draw

    @eqx.filter_jit
    def reverse(self, draws: Float[Array, "draws dim"]) -> Float[Array, "draws dim"]:
        f = jax.vmap(self.__reverse, (0, None))
        return f(draws, self.transform_params())

    def __forward_and_adjust(self, draw: Float[Array, " dim"], params: dict[str, Array]) -> Tuple[Float[Array, " dim"], Scalar]:
        assert len(draw.shape) == 1

        # Extract parameters
        input_knots: Float[Array, "dim n_bins+1"] = params["input_knots"]
        output_knots: Float[Array, "dim n_bins+1"] = params["output_knots"]
        lam: Float[Array, "dim n_bins"] = params["lam"]
        derivatives: Float[Array, "dim n_bins+1"] = params["derivatives"]

        # Match each element to its respective bin
        idx = jax.vmap(lambda x, k: jnp.searchsorted(k, x) - 1)(draw, input_knots)
        idx = jnp.clip(idx, 0, self.n_bins - 1)

        # Left of draw
        xk_lhs = gather(input_knots, idx)
        yk_lhs = gather(output_knots, idx)
        d_lhs  = gather(derivatives, idx)

        # Right of draw
        xk_rhs = gather(input_knots, idx + 1)
        yk_rhs = gather(output_knots, idx + 1)
        d_rhs  = gather(derivatives, idx + 1)

        # Relative midpoint for selected bin
        lam = gather(lam, idx)

        # Compute log-Jacobian adjustment and forward transformation ----
        output_spline, log_deriv = lr_forward(
            draw,
            xk_lhs,
            xk_rhs,
            yk_lhs,
            yk_rhs,
            d_lhs,
            d_rhs,
            lam
        )

        # Grab bounds
        llhs_ik = input_knots[:, 0]
        rrhs_ik = input_knots[:, -1]

        # Compute log-Jacobian adjustment
        log_jac = jnp.where(
            draw < llhs_ik,
            0.0,
            jnp.where(draw > rrhs_ik, 0.0, log_deriv)
        ).sum()

        # Compute forward transformation
        draw = jnp.where(draw < llhs_ik, draw, jnp.where(draw > rrhs_ik, draw, output_spline))

        assert log_jac.shape == ()

        return draw, log_jac

    @eqx.filter_jit
    def forward_and_adjust(self, draws: Float[Array, "draws dim"]) -> Tuple[Float[Array, "draws dim"], Scalar]:
        f = jax.vmap(self.__forward_and_adjust, (0, None))
        return f(draws, self.transform_params())

    def __reverse_and_adjust(self, draw: Float[Array, " dim"], params: dict[str, Array]) -> Tuple[Float[Array, " dim"], Scalar]:
        assert len(draw.shape) == 1

        # Extract parameters
        input_knots: Float[Array, "dim n_bins+1"] = params["input_knots"]
        output_knots: Float[Array, "dim n_bins+1"] = params["output_knots"]
        lam: Float[Array, "dim n_bins"] = params["lam"]
        derivatives: Float[Array, "dim n_bins+1"] = params["derivatives"]

        # Match each element to its respective bin
        idx = jax.vmap(lambda y, k: jnp.searchsorted(k, y) - 1)(draw, output_knots)
        idx = jnp.clip(idx, 0, self.n_bins - 1)

        # Left of draw
        xk_lhs = gather(input_knots, idx)
        yk_lhs = gather(output_knots, idx)
        d_lhs  = gather(derivatives, idx)

        # Right of draw
        xk_rhs = gather(input_knots, idx + 1)
        yk_rhs = gather(output_knots, idx + 1)
        d_rhs  = gather(derivatives, idx + 1)

        # Relative midpoint for selected bin
        lam = gather(lam, idx)

        # Compute log-Jacobian adjustment and reverse transformation ----
        input_spline, log_deriv = lr_reverse(
            draw, xk_lhs, xk_rhs, yk_lhs, yk_rhs, d_lhs, d_rhs, lam
        )

        # Grab bounds
        llhs_ok = output_knots[:, 0]
        rrhs_ok = output_knots[:, -1]

        # Compute log-Jacobian adjustment
        log_jac = jnp.where(
            draw < llhs_ok,
            0.0,
            jnp.where(draw > rrhs_ok, 0.0, log_deriv)
        ).sum()

        # Compute reverse transformation
        draw = jnp.where(draw < llhs_ok, draw, jnp.where(draw > rrhs_ok, draw, input_spline))

        return draw, log_jac

    @eqx.filter_jit
    def reverse_and_adjust(self, draws: Float[Array, "draws dim"]) -> Tuple[Float[Array, "draws dim"], Scalar]:
        f = jax.vmap(self.__reverse_and_adjust, (0, None))
        return f(draws, self.transform_params())

class LinearRationalSpline(FlowSpec):
    """
    A specification for a modified linear rational splines flow.

    Definition:
        The transformation $T(z)$ is defined as:

        $$
            T(\\tilde{z}) = \\begin{cases}
            y_0 + \\delta_0 (\\tilde{z} - x_0)(1 + |\\tilde{z} - x_0|)^\\alpha & z < x_0 \\\\
            \\text{LR}_b(\\tilde{z}) & x_b \\leq \\tilde{z} < x_{b+1} \\\\
            \\tilde{z} & \\tilde{z} > x_K
            \\end{cases}
        $$

        Where for each dimension $d \\in \\{1, \\dots, D\\}$ we centre the
        spline at $\\tilde{z} = z - s$, the knots are denoted by
        $\\{x_k, y_k\\}_{k=0}^K$ and derivatives by $\\{\\delta_k\\}_{k=0}^K$,
        with shift $s$. The linear-rational segment for bin $b$ is denoted by
        $\\text{LR}_b(z)$, and is parameterized by its knots
        $(x_b, y_b), (x_{b+1}, y_{b+1})$, derivatives at each knot
        $\\delta_b, \\delta_{b+1}$, and midpoint weight $\\lambda_b$
        controlling the curvature.

    Attributes:
        n_bins: The number of bins covering the boundary.
        key: The PRNG key used to generate the diagonal affine flow layer.
    """
    n_bins: int
    key: PRNGKeyArray

    def __init__(
        self,
        n_bins: int = 7,
        key: PRNGKeyArray = jr.key(0)
    ):
        self.n_bins = n_bins
        self.key = key

    def construct(self, dim: int) -> LinearRationalSplineLayer:
        return LinearRationalSplineLayer(dim, self.n_bins, self.key)
