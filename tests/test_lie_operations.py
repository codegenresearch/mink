"""Tests for general operation definitions."""

from typing import Type

import numpy as np
from absl.testing import absltest, parameterized

from mink.lie.base import MatrixLieGroup
from mink.lie.se3 import SE3
from mink.lie.so3 import SO3

from .utils import assert_transforms_close


@parameterized.named_parameters(
    ("SO3", SO3),
    ("SE3", SE3),
)
class TestOperations(parameterized.TestCase):
    def test_inverse_bijective(self, group: Type[MatrixLieGroup]):
        """Check inverse of inverse."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, transform.inverse().inverse())

    def test_matrix_bijective(self, group: Type[MatrixLieGroup]):
        """Check that we can convert to and from matrices."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, group.from_matrix(transform.as_matrix()))

    def test_log_exp_bijective(self, group: Type[MatrixLieGroup]):
        """Check 1-to-1 mapping for log <=> exp operations."""
        transform = group.sample_uniform()

        tangent = transform.log()
        self.assertEqual(tangent.shape, (group.tangent_dim,))

        exp_transform = group.exp(tangent)
        assert_transforms_close(transform, exp_transform)
        np.testing.assert_allclose(tangent, exp_transform.log())

    def test_adjoint(self, group: Type[MatrixLieGroup]):
        """Check adjoint operation."""
        transform = group.sample_uniform()
        omega = np.random.randn(group.tangent_dim)
        assert_transforms_close(
            transform @ group.exp(omega),
            group.exp(transform.adjoint() @ omega) @ transform,
        )

    def test_rminus(self, group: Type[MatrixLieGroup]):
        """Check rminus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a.inverse() @ T_b
        np.testing.assert_allclose(T_b.rminus(T_a), T_c.log())

    def test_lminus(self, group: Type[MatrixLieGroup]):
        """Check lminus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        np.testing.assert_allclose(T_a.lminus(T_b), (T_a @ T_b.inverse()).log())

    def test_rplus(self, group: Type[MatrixLieGroup]):
        """Check rplus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a.inverse() @ T_b
        assert_transforms_close(T_a.rplus(T_c.log()), T_b)

    def test_lplus(self, group: Type[MatrixLieGroup]):
        """Check lplus operation."""
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a @ T_b.inverse()
        assert_transforms_close(T_b.lplus(T_c.log()), T_a)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        """Check jlog operation."""
        state = group.sample_uniform()
        w = np.random.rand(state.tangent_dim) * 1e-4
        state_pert = state.plus(w).log()
        state_lin = state.log() + state.jlog() @ w
        np.testing.assert_allclose(state_pert, state_lin, atol=1e-7)

    def test_rplus_with_zero_tangent(self, group: Type[MatrixLieGroup]):
        """Check rplus with zero tangent."""
        transform = group.sample_uniform()
        zero_tangent = np.zeros(transform.tangent_dim)
        assert_transforms_close(transform.rplus(zero_tangent), transform)

    def test_lplus_with_zero_tangent(self, group: Type[MatrixLieGroup]):
        """Check lplus with zero tangent."""
        transform = group.sample_uniform()
        zero_tangent = np.zeros(transform.tangent_dim)
        assert_transforms_close(transform.lplus(zero_tangent), transform)

    def test_rminus_with_identity(self, group: Type[MatrixLieGroup]):
        """Check rminus with identity."""
        transform = group.sample_uniform()
        identity = group.identity()
        np.testing.assert_allclose(transform.rminus(identity), np.zeros(transform.tangent_dim))

    def test_lminus_with_identity(self, group: Type[MatrixLieGroup]):
        """Check lminus with identity."""
        transform = group.sample_uniform()
        identity = group.identity()
        np.testing.assert_allclose(transform.lminus(identity), np.zeros(transform.tangent_dim))


class TestGroupSpecificOperations(absltest.TestCase):
    """Group specific tests."""

    def test_se3_translation(self):
        """Check SE3 translation."""
        T = SE3.sample_uniform()
        translation = T.as_matrix()[:3, 3]
        assert_transforms_close(T, SE3.from_translation(translation))

    def test_se3_rotation(self):
        """Check SE3 rotation."""
        T = SE3.sample_uniform()
        rotation_matrix = T.as_matrix()[:3, :3]
        assert_transforms_close(T, SE3.from_rotation_matrix(rotation_matrix))

    @parameterized.named_parameters(
        ("SO3", SO3),
    )
    def test_so3_rpy_bijective(self, group: Type[MatrixLieGroup]):
        """Check SO3 RPY bijectivity."""
        T = SO3.sample_uniform()
        assert_transforms_close(T, SO3.from_rpy_radians(*T.as_rpy_radians()))

    def test_so3_copy(self):
        """Check SO3 copy method."""
        T = SO3.sample_uniform()
        T_copy = T.copy()
        assert_transforms_close(T, T_copy)

    def test_se3_copy(self):
        """Check SE3 copy method."""
        T = SE3.sample_uniform()
        T_copy = T.copy()
        assert_transforms_close(T, T_copy)

    def test_invalid_log_exp_so3(self):
        """Check that exp raises error if invalid shape."""
        with self.assertRaises(ValueError):
            SO3.exp(np.random.rand(4))  # Invalid tangent vector size

    def test_invalid_matrix_conversion_so3(self):
        """Check that from_matrix raises error if invalid matrix size."""
        with self.assertRaises(ValueError):
            SO3.from_matrix(np.random.rand(3, 3))  # Invalid matrix size

    def test_invalid_matrix_conversion_se3(self):
        """Check that from_matrix raises error if invalid matrix size."""
        with self.assertRaises(ValueError):
            SE3.from_matrix(np.random.rand(4, 4))  # Invalid matrix size


if __name__ == "__main__":
    absltest.main()