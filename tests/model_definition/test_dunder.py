from typing import List

import jax.numpy as jnp
from jaxtyping import Array

from bayinx import Model, define
from bayinx.dists import Normal
from bayinx.nodes import Continuous, Observed


class MyModel(Model, init=False):
    x: Continuous[Array] = define(shape = (2,2), init = jnp.ones((2,2)))
    y: Observed[Array] = define(shape = (2,2), init = jnp.ones((2,2)))
    z: Observed[List[Array]] = define(init = [jnp.array(-1.0), jnp.array(0.0), jnp.array(1.0)])


    def model(self, target):
        add_node = self.x + self.y
        sub_node = self.x - self.y
        mul_node = self.x * self.y
        div_node = self.x / self.y
        matmul_node = self.x @ self.y


        add_node << Normal(0.0, 1.0)
        sub_node << Normal(0.0, 1.0)
        mul_node << Normal(0.0, 1.0)
        div_node << Normal(0.0, 1.0)
        matmul_node << Normal(0.0, 1.0)

        for node in self.z:
            node << Normal(0.0, 1.0)

        return target


def test_dunder():
    # Construct model
    model = MyModel()

    assert not jnp.isnan(model())
