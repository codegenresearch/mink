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
    assert tangent.shape == (group.tangent_dim,)

    exp_transform = group.exp(tangent)
    assert_transforms_close(transform, exp_transform)
    np.testing.assert_allclose(tangent, exp_transform.log())


def check_adjoint(transform: MatrixLieGroup, group: Type[MatrixLieGroup]):
    omega = np.random.randn(group.tangent_dim)
    assert_transforms_close(
        transform @ group.exp(omega),
        group.exp(transform.adjoint() @ omega) @ transform,
    )


def check_rminus(T_a: MatrixLieGroup, T_b: MatrixLieGroup):
    T_c = T_a.inverse() @ T_b
    np.testing.assert_allclose(T_b.rminus(T_a), T_c.log())


def check_lminus(T_a: MatrixLieGroup, T_b: MatrixLieGroup):
    np.testing.assert_allclose(T_a.lminus(T_b), (T_a @ T_b.inverse()).log())


def check_rplus(T_a: MatrixLieGroup, T_b: MatrixLieGroup):
    T_c = T_a.inverse() @ T_b
    assert_transforms_close(T_a.rplus(T_c.log()), T_b)


def check_lplus(T_a: MatrixLieGroup, T_b: MatrixLieGroup):
    T_c = T_a @ T_b.inverse()
    assert_transforms_close(T_b.lplus(T_c.log()), T_a)


def check_jlog(state: MatrixLieGroup):
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
        """Check inverse of inverse."""
        transform = group.sample_uniform()
        check_inverse_bijective(transform)

    def test_matrix_bijective(self, group: Type[MatrixLieGroup]):
        """Check that we can convert to and from matrices."""
        transform = group.sample_uniform()
        check_matrix_bijective(transform, group)

    def test_log_exp_bijective(self, group: Type[MatrixLieGroup]):
        """Check 1-to-1 mapping for log <=> exp operations."""
        transform = group.sample_uniform()
        check_log_exp_bijective(transform, group)

    def test_adjoint(self, group: Type[MatrixLieGroup]):
        """Check adjoint operation."""
        transform = group.sample_uniform()
        check_adjoint(transform, group)

    def test_rminus(self, group: Type[MatrixLieGroup]):
        """Check rminus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_rminus(T_a, T_b)

    def test_lminus(self, group: Type[MatrixLieGroup]):
        """Check lminus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_lminus(T_a, T_b)

    def test_rplus(self, group: Type[MatrixLieGroup]):
        """Check rplus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_rplus(T_a, T_b)

    def test_lplus(self, group: Type[MatrixLieGroup]):
        """Check lplus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        check_lplus(T_a, T_b)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        """Check jlog operation."""
        state = group.sample_uniform()
        check_jlog(state)


class TestGroupSpecificOperations(absltest.TestCase):
    """Group specific tests."""

    @parameterized.named_parameters(
        ("SO3", SO3),
        ("SE3", SE3),
    )
    def test_rpy_bijective(self, group: Type[MatrixLieGroup]):
        """Check RPY conversion is bijective for SO3."""
        if group is SO3:
            T = group.sample_uniform()
            assert_transforms_close(T, group.from_rpy_radians(*T.as_rpy_radians()))

    def test_rpy_bijective_random(self):
        """Check RPY conversion is bijective for SO3 with random samples."""
        T = SO3.sample_uniform()
        rpy = T.as_rpy_radians()
        T_reconstructed = SO3.from_rpy_radians(*rpy)
        assert_transforms_close(T, T_reconstructed)

    def test_rpy_bijective_edge_cases(self):
        """Check RPY conversion is bijective for SO3 with edge cases."""
        edge_cases = [
            (0, 0, 0),
            (np.pi/2, 0, 0),
            (0, np.pi/2, 0),
            (0, 0, np.pi/2),
            (np.pi, 0, 0),
            (0, np.pi, 0),
            (0, 0, np.pi),
            (np.pi, np.pi, np.pi),
        ]
        for r, p, y in edge_cases:
            T = SO3.from_rpy_radians(r, p, y)
            rpy = T.as_rpy_radians()
            T_reconstructed = SO3.from_rpy_radians(*rpy)
            assert_transforms_close(T, T_reconstructed)

    def test_invalid_shape_log_exp(self):
        """Check that log and exp raise errors for invalid shapes."""
        T = SO3.sample_uniform()
        invalid_tangent = np.random.rand(T.tangent_dim + 1)
        with self.assertRaises(ValueError):
            T.exp(invalid_tangent)
        with self.assertRaises(ValueError):
            T.log(invalid_tangent)

    def test_copy_operation(self):
        """Check that copying a transformation results in an identical transformation."""
        T = SO3.sample_uniform()
        T_copy = T.copy()
        assert_transforms_close(T, T_copy)


if __name__ == "__main__":
    absltest.main()