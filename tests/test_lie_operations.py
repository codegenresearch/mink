"""Tests for general operation definitions."""

from typing import Type

import numpy as np
from absl.testing import absltest, parameterized

from mink.lie.base import MatrixLieGroup
from mink.lie.se3 import SE3
from mink.lie.so3 import SO3

from .utils import assert_transforms_close


def check_inverse_bijective(transform: MatrixLieGroup):
    """Check inverse of inverse."""
    assert_transforms_close(transform, transform.inverse().inverse())


def check_matrix_bijective(transform: MatrixLieGroup, group: Type[MatrixLieGroup]):
    """Check that we can convert to and from matrices."""
    assert_transforms_close(transform, group.from_matrix(transform.as_matrix()))


def check_log_exp_bijective(transform: MatrixLieGroup, group: Type[MatrixLieGroup]):
    """Check 1-to-1 mapping for log <=> exp operations."""
    tangent = transform.log()
    self.assertEqual(tangent.shape, (group.tangent_dim,))

    exp_transform = group.exp(tangent)
    assert_transforms_close(transform, exp_transform)
    np.testing.assert_allclose(tangent, exp_transform.log())


def check_adjoint(transform: MatrixLieGroup, group: Type[MatrixLieGroup]):
    """Check adjoint operation."""
    omega = np.random.randn(group.tangent_dim)
    assert_transforms_close(
        transform @ group.exp(omega),
        group.exp(transform.adjoint() @ omega) @ transform,
    )


def check_rminus(T_a: MatrixLieGroup, T_b: MatrixLieGroup):
    """Check rminus operation."""
    T_c = T_a.inverse() @ T_b
    np.testing.assert_allclose(T_b.rminus(T_a), T_c.log())


def check_lminus(T_a: MatrixLieGroup, T_b: MatrixLieGroup):
    """Check lminus operation."""
    np.testing.assert_allclose(T_a.lminus(T_b), (T_a @ T_b.inverse()).log())


def check_rplus(T_a: MatrixLieGroup, T_b: MatrixLieGroup):
    """Check rplus operation."""
    T_c = T_a.inverse() @ T_b
    assert_transforms_close(T_a.rplus(T_c.log()), T_b)


def check_lplus(T_a: MatrixLieGroup, T_b: MatrixLieGroup):
    """Check lplus operation."""
    T_c = T_a @ T_b.inverse()
    assert_transforms_close(T_b.lplus(T_c.log()), T_a)


def check_jlog(state: MatrixLieGroup):
    """Check jlog operation."""
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
        """Test that the inverse of the inverse of a transform is the original transform."""
        transform = group.sample_uniform()
        check_inverse_bijective(transform)

    def test_matrix_bijective(self, group: Type[MatrixLieGroup]):
        """Test that converting a transform to a matrix and back results in the original transform."""
        transform = group.sample_uniform()
        check_matrix_bijective(transform, group)

    def test_log_exp_bijective(self, group: Type[MatrixLieGroup]):
        """Test that the log and exp operations are bijective."""
        transform = group.sample_uniform()
        check_log_exp_bijective(transform, group)

    def test_adjoint(self, group: Type[MatrixLieGroup]):
        """Test the adjoint operation."""
        transform = group.sample_uniform()
        check_adjoint(transform, group)

    def test_rminus(self, group: Type[MatrixLieGroup]):
        """Test the rminus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_rminus(T_a, T_b)

    def test_lminus(self, group: Type[MatrixLieGroup]):
        """Test the lminus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_lminus(T_a, T_b)

    def test_rplus(self, group: Type[MatrixLieGroup]):
        """Test the rplus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_rplus(T_a, T_b)

    def test_lplus(self, group: Type[MatrixLieGroup]):
        """Test the lplus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_lplus(T_a, T_b)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        """Test the jlog operation."""
        state = group.sample_uniform()
        check_jlog(state)

    def test_rplus_with_zero_tangent(self, group: Type[MatrixLieGroup]):
        """Test that rplus with a zero tangent vector returns the original transform."""
        transform = group.sample_uniform()
        zero_tangent = np.zeros(transform.tangent_dim)
        assert_transforms_close(transform.rplus(zero_tangent), transform)

    def test_lplus_with_zero_tangent(self, group: Type[MatrixLieGroup]):
        """Test that lplus with a zero tangent vector returns the original transform."""
        transform = group.sample_uniform()
        zero_tangent = np.zeros(transform.tangent_dim)
        assert_transforms_close(transform.lplus(zero_tangent), transform)

    def test_rminus_with_identity(self, group: Type[MatrixLieGroup]):
        """Test that rminus with the identity transform returns a zero tangent vector."""
        transform = group.sample_uniform()
        identity = group.identity()
        np.testing.assert_allclose(transform.rminus(identity), np.zeros(transform.tangent_dim))

    def test_lminus_with_identity(self, group: Type[MatrixLieGroup]):
        """Test that lminus with the identity transform returns a zero tangent vector."""
        transform = group.sample_uniform()
        identity = group.identity()
        np.testing.assert_allclose(transform.lminus(identity), np.zeros(transform.tangent_dim))

    def test_se3_translation(self):
        """Test that SE3 can be constructed from a translation vector."""
        T = SE3.sample_uniform()
        translation = T.as_matrix()[:3, 3]
        assert_transforms_close(T, SE3.from_translation(translation))

    def test_se3_rotation(self):
        """Test that SE3 can be constructed from a rotation matrix."""
        T = SE3.sample_uniform()
        rotation_matrix = T.as_matrix()[:3, :3]
        assert_transforms_close(T, SE3.from_rotation_matrix(rotation_matrix))


if __name__ == "__main__":
    absltest.main()