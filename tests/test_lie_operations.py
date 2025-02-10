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
        """Verify inverse operation is bijective."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, transform.inverse().inverse())

    def test_matrix_conversion_bijective(self, group: Type[MatrixLieGroup]):
        """Ensure matrix conversion is bijective."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, group.from_matrix(transform.as_matrix()))

    def test_log_exp_bijective(self, group: Type[MatrixLieGroup]):
        """Validate log and exp operations are bijective."""
        transform = group.sample_uniform()

        tangent = transform.log()
        self.assertEqual(tangent.shape, (group.tangent_dim,))

        exp_transform = group.exp(tangent)
        assert_transforms_close(transform, exp_transform)
        np.testing.assert_allclose(tangent, exp_transform.log())

    def test_adjoint(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        omega = np.random.randn(group.tangent_dim)
        assert_transforms_close(
            transform @ group.exp(omega),
            group.exp(transform.adjoint() @ omega) @ transform,
        )

    def test_rminus(self, group: Type[MatrixLieGroup]):
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a.inverse() @ transform_b
        np.testing.assert_allclose(transform_b.rminus(transform_a), transform_c.log())

    def test_lminus(self, group: Type[MatrixLieGroup]):
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        np.testing.assert_allclose(transform_a.lminus(transform_b), (transform_a @ transform_b.inverse()).log())

    def test_rplus(self, group: Type[MatrixLieGroup]):
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a.inverse() @ transform_b
        assert_transforms_close(transform_a.rplus(transform_c.log()), transform_b)

    def test_lplus(self, group: Type[MatrixLieGroup]):
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a @ transform_b.inverse()
        assert_transforms_close(transform_b.lplus(transform_c.log()), transform_a)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        w = np.random.rand(transform.tangent_dim) * 1e-4
        transform_pert = transform.plus(w).log()
        transform_lin = transform.log() + transform.jlog() @ w
        np.testing.assert_allclose(transform_pert, transform_lin, atol=1e-7)


class TestSpecificOperations(absltest.TestCase):
    """Tests specific to individual groups."""

    def test_so3_rpy_conversion(self):
        transform = SO3.sample_uniform()
        assert_transforms_close(transform, SO3.from_rpy_radians(*transform.as_rpy_radians()))

    def test_se3_translation_conversion(self):
        transform = SE3.sample_uniform()
        translation = transform.as_matrix()[:3, 3]
        assert_transforms_close(transform, SE3.from_translation(translation))

    def test_se3_rotation_conversion(self):
        transform = SE3.sample_uniform()
        rotation = SO3(transform.as_matrix()[:3, :3])
        assert_transforms_close(transform, SE3.from_rotation(rotation))

    def test_so3_invalid_rpy(self):
        with self.assertRaises(ValueError):
            SO3.from_rpy_radians(10, 10, 10)  # Invalid RPY angles

    def test_se3_invalid_translation(self):
        with self.assertRaises(ValueError):
            SE3.from_translation(np.array([1, 2]))  # Invalid translation vector

    def test_se3_invalid_rotation(self):
        with self.assertRaises(ValueError):
            SE3.from_rotation(SO3(np.eye(2)))  # Invalid rotation matrix

    def test_se3_from_matrix(self):
        transform = SE3.sample_uniform()
        matrix = transform.as_matrix()
        assert_transforms_close(transform, SE3.from_matrix(matrix))

    def test_so3_from_matrix(self):
        transform = SO3.sample_uniform()
        matrix = transform.as_matrix()
        assert_transforms_close(transform, SO3.from_matrix(matrix))


if __name__ == "__main__":
    absltest.main()