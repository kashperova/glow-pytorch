import pytest

from fixtures.config import TestConfig
from model.actnorm import ActNorm
from model.affine_coupling import AffineCoupling
from model.flow import Flow
from model.flow_block import FlowBlock
from model.glow import Glow
from model.invert_conv import InvertConv


@pytest.fixture(scope="function")
def act_norm():
    return ActNorm(in_ch=TestConfig.in_ch)


@pytest.fixture(scope="function")
def invert_conv():
    return InvertConv(in_ch=TestConfig.in_ch)


@pytest.fixture(scope="function")
def affine_coupling():
    return AffineCoupling(
        in_ch=TestConfig.in_ch, hidden_ch=TestConfig.coupling_hidden_ch
    )


@pytest.fixture(scope="function")
def flow():
    return Flow(
        in_ch=TestConfig.in_ch, coupling_hidden_ch=TestConfig.coupling_hidden_ch
    )


@pytest.fixture(scope="function")
def flow_block():
    return FlowBlock(
        in_ch=TestConfig.in_ch,
        n_flows=TestConfig.n_flows,
        squeeze_factor=TestConfig.squeeze_factor,
        coupling_hidden_ch=TestConfig.coupling_hidden_ch,
    )


@pytest.fixture(scope="function")
def last_flow_block():
    return FlowBlock(
        in_ch=TestConfig.in_ch,
        n_flows=TestConfig.n_flows,
        squeeze_factor=TestConfig.squeeze_factor,
        coupling_hidden_ch=TestConfig.coupling_hidden_ch,
        split=False,
    )


@pytest.fixture(scope="function")
def glow():
    return Glow(
        in_ch=TestConfig.in_ch,
        num_blocks=TestConfig.num_blocks,
        n_flows=TestConfig.n_flows,
        squeeze_factor=TestConfig.squeeze_factor,
        coupling_hidden_ch=TestConfig.coupling_hidden_ch,
    )
