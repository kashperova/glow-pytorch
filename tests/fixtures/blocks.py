import pytest

from fixtures.config import TestConfig
from model.actnorm import ActNorm
from model.affine_coupling import AffineCoupling
from model.flow import Flow
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
