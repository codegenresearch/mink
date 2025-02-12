"""Tests for general operation definitions."""

from typing import Type

import numpy as np
from absl.testing import absltest, parameterized

from mink.lie.base import MatrixLieGroup
from mink.lie.se3 import SE3
from mink.lie.so3 import SO3

from .utils import assert_transforms_close

def check_inverse_bijective(transform: MatrixLieGroup):
    """Check that the inverse of the inverse is the original transform."""
    assert_transforms_close(transform, transform.inverse().inverse())

def check_matrix_bijective(transform: MatrixLieGroup, group: Type[MatrixLieGroup]):
    """Check that converting to and from matrices is bijective."""
    assert_transforms_close(transform, group.from_matrix(transform.as_matrix()))

def check_log_exp_bijective(transform: MatrixLieGroup, group: Type[MatrixLieGroup]):
    """Check that the log and exp operations are bijective."""
    tangent = transform.log()
    assert tangent.shape == (group.tangent_dim,)

    exp_transform = group.exp(tangent)
    assert_transforms_close(transform, exp_transform)
    np.testing.assert_allclose(tangent, exp_transform.log())

def check_adjoint(transform: MatrixLieGroup, group: Type[MatrixLieGroup]):
    """Check that the adjoint operation satisfies the expected property."""
    omega = np.random.randn(group.tangent_dim)
    assert_transforms_close(
        transform @ group.exp(omega),
        group.exp(transform.adjoint() @ omega) @ transform,
    )

def check_rminus(transform_a: MatrixLieGroup, transform_b: MatrixLieGroup):
    """Check the correctness of the rminus operation."""
    transform_c = transform_a.inverse() @ transform_b
    np.testing.assert_allclose(transform_b.rminus(transform_a), transform_c.log())

def check_lminus(transform_a: MatrixLieGroup, transform_b: MatrixLieGroup):
    """Check the correctness of the lminus operation."""
    np.testing.assert_allclose(transform_a.lminus(transform_b), (transform_a @ transform_b.inverse()).log())

def check_rplus(transform_a: MatrixLieGroup, transform_b: MatrixLieGroup):
    """Check the correctness of the rplus operation."""
    transform_c = transform_a.inverse() @ transform_b
    assert_transforms_close(transform_a.rplus(transform_c.log()), transform_b)

def check_lplus(transform_a: MatrixLieGroup, transform_b: MatrixLieGroup):
    """Check the correctness of the lplus operation."""
    transform_c = transform_a @ transform_b.inverse()
    assert_transforms_close(transform_b.lplus(transform_c.log()), transform_a)

def check_jlog(state: MatrixLieGroup):
    """Check the correctness of the jlog operation."""
    w = np.random.rand(state.tangent_dim) * 1e-4
    state_pert = state.plus(w).log()
    state_lin = state.log() + state.jlog() @ w
    np.testing.assert_allclose(state_pert, state_lin, atol=1e-7)

@parameterized.named_parameters(
    ("SO3", SO3),
    ("SE3", SE3),
)
class TestOperations(parameterized.TestCase):
    def test_inverse_bijective(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        check_inverse_bijective(transform)

    def test_matrix_bijective(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        check_matrix_bijective(transform, group)

    def test_log_exp_bijective(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        check_log_exp_bijective(transform, group)

    def test_adjoint(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        check_adjoint(transform, group)

    def test_rminus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_rminus(T_a, T_b)

    def test_lminus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_lminus(T_a, T_b)

    def test_rplus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_rplus(T_a, T_b)

    def test_lplus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_lplus(T_a, T_b)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        state = group.sample_uniform()
        check_jlog(state)

    def test_exp_ljac(self, group: Type[MatrixLieGroup]):
        """Check that the exponential map and left Jacobian are consistent."""
        transform = group.sample_uniform()
        tangent = np.random.randn(group.tangent_dim)
        exp_transform = group.exp(tangent)
        ljac = group.ljac(tangent)
        np.testing.assert_allclose(exp_transform.parameters(), (np.eye(group.parameters_dim) + ljac @ tangent) @ transform.parameters())

    def test_exp_rjac(self, group: Type[MatrixLieGroup]):
        """Check that the exponential map and right Jacobian are consistent."""
        transform = group.sample_uniform()
        tangent = np.random.randn(group.tangent_dim)
        exp_transform = group.exp(tangent)
        rjac = group.rjac(tangent)
        np.testing.assert_allclose(exp_transform.parameters(), transform.parameters() @ (np.eye(group.parameters_dim) + rjac @ tangent))

class TestGroupSpecificOperations(absltest.TestCase):
    """Group specific tests."""

    def test_so3_rpy_bijective(self):
        T = SO3.sample_uniform()
        assert_transforms_close(T, SO3.from_rpy_radians(*T.as_rpy_radians()))

    def test_se3_translation(self):
        T = SE3.sample_uniform()
        translation = T.as_matrix()[:3, 3]
        assert np.allclose(translation, T.translation)

    def test_se3_rotation(self):
        T = SE3.sample_uniform()
        rotation = T.as_matrix()[:3, :3]
        assert np.allclose(rotation, T.rotation.as_matrix())

if __name__ == "__main__":
    absltest.main()