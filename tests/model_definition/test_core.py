# pyright: reportUnusedExpression=false
from typing import Dict, List

import jax.numpy as jnp
from jaxtyping import Array

from bayinx import Model, define
from bayinx.dists import Normal
from bayinx.nodes import Continuous, Observed


# Define model
class MyModel(Model, init=False):
    x: Continuous[Array] = define() # shows up in __init__ method, typed as 'Array'
    y: Observed[Array] = define() # shows up in __init__ method, typed as 'Array'
    misc_1: Observed[str] = define(init = "hi")
    misc_2: Observed[List[str]] = define(init = ["ciao", "goodbye"])
    misc_3: Observed[Dict[str, str]] = define(init = {'key': 'value'})

    def model(self, target):
        self.y << Normal(self.x, 1.0)

        return target


def test_init():
    """
    Test aspects of initialization.
    """
    model = MyModel(x = jnp.array(0.0), y = jnp.array([0.0, 0.0, 0.0]))

    # Test filter specification
    assert model.filter_spec.x.obj # Stochastic nodes should be included
    assert not model.filter_spec.y.obj # Observed nodes should be excluded
    assert not model.filter_spec.misc_1.obj # Regardless of type
    assert not model.filter_spec.misc_2.obj[0] # But should be PyTrees internally

    # Test (log unnormalized) posterior evaluation
    assert abs(model() - jnp.array(-2.756815599614018453)) < 1e-6 # 3 * log P_Z(0)

def test_subset():
    """
    Test hard subsetting.
    """
    model = MyModel(x = jnp.array(0.0), y = jnp.array([0.0, 0.0, 0.0]))

    assert model.misc_2[0].obj == "ciao"
    assert model.misc_3['key'].obj == "value"
