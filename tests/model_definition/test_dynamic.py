# pyright: reportUnusedExpression=false

import jax.numpy as jnp
from jaxtyping import Array

from bayinx import Model, define
from bayinx.dists import Normal
from bayinx.nodes import Continuous, Observed


class SimpleDynamic(Model, init=False):
    mu: Continuous[Array] = define(shape = ())
    y: Observed[Array] = define()


    def model(self, target):
        self.y << Normal(self.mu, 1.0)

        return target


def test_simple():
    y: Array = jnp.array([-1.0, 0.0, 1.0])

    # Construct model
    model = SimpleDynamic(y = y)
    assert all(model.y.obj == jnp.array([-1.0, 0.0, 1.0]))


class ShapedDynamic(Model, init=False):
    mus: Continuous[Array] = define(shape = ('k'))
    y: Observed[Array] = define(shape = ('n', 'k'))

    def model(self, target):
        self.y << Normal(self.mus, 1.0)

        return target

def test_shaped():
    y: Array = jnp.tile(jnp.array([-1.0, 0.0, 1.0]), (10, 1))
    #bad_y: Array = jnp.tile(jnp.array([-1.0, 1.0]), (10, 1))

    model = ShapedDynamic(k = 3, n = 10, y = y)
    assert model.mus.obj.shape == (3,)
    assert model.y.obj.shape == (10,3)

class PackedShapeDynamic(Model, init=False):
    x: Continuous[Array] = define(shape = 'shape')
    y: Continuous[Array] = define(shape = ('m', 'n', 'shape'))

    def model(self, target):
        self.x << Normal(0.0, 1.0)

        return target

def test_packedshape():
    model = PackedShapeDynamic(shape = (3,2,1), m = 5, n = 4)
    assert model.x.obj.shape == (3,2,1)
    assert model.y.obj.shape == (5,4,3,2,1)
