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
    def test_inverse(self, group: Type[MatrixLieGroup]):
        """Verify inverse operation is bijective."""
        T = group.sample_uniform()
        assert_transforms_close(T, T.inverse().inverse())

    def test_matrix_conversion(self, group: Type[MatrixLieGroup]):
        """Ensure matrix conversion is bijective."""
        T = group.sample_uniform()
        assert_transforms_close(T, group.from_matrix(T.as_matrix()))

    def test_log_exp(self, group: Type[MatrixLieGroup]):
        """Validate log and exp operations are bijective."""
        T = group.sample_uniform()

        tangent = T.log()
        self.assertEqual(tangent.shape, (group.tangent_dim,))

        exp_T = group.exp(tangent)
        assert_transforms_close(T, exp_T)
        np.testing.assert_allclose(tangent, exp_T.log())

    def test_adjoint(self, group: Type[MatrixLieGroup]):
        T = group.sample_uniform()
        omega = np.random.randn(group.tangent_dim)
        assert_transforms_close(
            T @ group.exp(omega),
            group.exp(T.adjoint() @ omega) @ T,
        )

    def test_rminus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a.inverse() @ T_b
        np.testing.assert_allclose(T_b.rminus(T_a), T_c.log())

    def test_lminus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        np.testing.assert_allclose(T_a.lminus(T_b), (T_a @ T_b.inverse()).log())

    def test_rplus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a.inverse() @ T_b
        assert_transforms_close(T_a.rplus(T_c.log()), T_b)

    def test_lplus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a @ T_b.inverse()
        assert_transforms_close(T_b.lplus(T_c.log()), T_a)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        T = group.sample_uniform()
        w = np.random.rand(T.tangent_dim) * 1e-4
        T_pert = T.plus(w).log()
        T_lin = T.log() + T.jlog() @ w
        np.testing.assert_allclose(T_pert, T_lin, atol=1e-7)


class TestSpecificOperations(absltest.TestCase):
    """Tests specific to individual groups."""

    def test_so3_rpy_conversion(self):
        T = SO3.sample_uniform()
        assert_transforms_close(T, SO3.from_rpy_radians(*T.as_rpy_radians()))

    def test_se3_translation(self):
        T = SE3.sample_uniform()
        translation = T.as_matrix()[:3, 3]
        assert_transforms_close(T, SE3.from_translation(translation))

    def test_se3_rotation(self):
        T = SE3.sample_uniform()
        rotation = T.as_matrix()[:3, :3]
        assert_transforms_close(T, SE3.from_rotation(rotation))

    def test_so3_invalid_rpy(self):
        with self.assertRaises(ValueError):
            SO3.from_rpy_radians(10, 10, 10)  # Invalid RPY angles

    def test_se3_invalid_translation(self):
        with self.assertRaises(ValueError):
            SE3.from_translation(np.array([1, 2]))  # Invalid translation vector

    def test_se3_invalid_rotation(self):
        with self.assertRaises(ValueError):
            SE3.from_rotation(np.eye(2))  # Invalid rotation matrix


if __name__ == "__main__":
    absltest.main()