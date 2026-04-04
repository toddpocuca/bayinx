import inspect

import jax
import jax.numpy as jnp
import pytest

import bayinx.flows as flows
from bayinx import Model, Posterior, define
from bayinx.core.flow import FlowSpec
from bayinx.dists import Normal
from bayinx.nodes import Continuous

jax.config.update("jax_enable_x64", True)

eps = jnp.finfo(jnp.array(0.0)).eps

# Filter the module to get only the classes you've imported/defined
flow_classes = list({
    obj for name, obj in vars(flows).items()
    if inspect.isclass(obj) and obj.__module__.startswith("bayinx.flows")
})

def init_flowspec(flowspec: type[FlowSpec]):
    sig = inspect.signature(flowspec.__init__)
    if "flip" in sig.parameters:
        return [flowspec(flip=True), flowspec(flip=False)] # type: ignore
    return [flowspec()]

class MyModel(Model):
    x: Continuous = define(shape = (4, ))

    def model(self, target):
        self.x << Normal(1.0, 2.0)


@pytest.mark.parametrize("flowspec", flow_classes)
def test_transformation(flowspec: type[FlowSpec]):
    # Check transformation's forward and reverse undo each other
    flow = flowspec().construct(2)
    draw = jnp.ones(2).reshape(1,2)

    assert (abs(flow.reverse(flow.forward(draw)) - draw) < 10*eps).all()
    assert (abs(flow.forward(flow.reverse(draw)) - draw) < 10*eps).all()
    assert abs(flow.forward_and_adjust(draw)[1] + flow.reverse_and_adjust(flow.forward(draw))[1]) < 10*eps
    assert abs(flow.reverse_and_adjust(draw)[1] + flow.forward_and_adjust(flow.reverse(draw))[1]) < 10*eps

@pytest.mark.parametrize("flowspec", flow_classes)
def test_normal_fit(flowspec):
    """
    Test that every flow can be instantiated and fits a simple distribution.
    """
    # Construct posterior
    posterior = Posterior(
        MyModel
    )
    posterior.configure(init_flowspec(flowspec))
    posterior.fit(stl = False)

    # Check samples
    x_draws = posterior.sample('x', int(1e6))
    assert (abs(x_draws.mean(0) - 1.0) / 1.0 < 1e-1).all()
    assert (abs(x_draws.var(0) - 4.0) / 4.0 < 1e-1).all()

@pytest.mark.parametrize("flowspec", flow_classes)
def test_stl_fit(flowspec):
    """
    Test that every flow can be instantiated and fits a simple distribution.
    """
    # Construct posterior
    posterior = Posterior(
        MyModel
    )
    posterior.configure(init_flowspec(flowspec))
    posterior.fit(stl = True)

    # Check samples
    x_draws = posterior.sample('x', int(1e6))
    assert (abs(x_draws.mean(0) - 1.0) / 1.0 < 1e-1).all()
    assert (abs(x_draws.var(0) - 4.0) / 4.0 < 1e-1).all()
