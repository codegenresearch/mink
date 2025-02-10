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
class TestGeneralOperations(parameterized.TestCase):
    def test_inverse_is_bijective(self, group: Type[MatrixLieGroup]):
        """Verify that applying inverse twice returns the original transform."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, transform.inverse().inverse())

    def test_matrix_conversion_is_bijective(self, group: Type[MatrixLieGroup]):
        """Ensure conversion to and from matrix representations is accurate."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, group.from_matrix(transform.as_matrix()))

    def test_log_exp_operations_are_bijective(self, group: Type[MatrixLieGroup]):
        """Validate the one-to-one mapping between log and exp operations."""
        transform = group.sample_uniform()

        tangent = transform.log()
        self.assertEqual(tangent.shape, (group.tangent_dim,))

        exp_transform = group.exp(tangent)
        assert_transforms_close(transform, exp_transform)
        np.testing.assert_allclose(tangent, exp_transform.log())

    def test_adjoint_operation(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        omega = np.random.randn(group.tangent_dim)
        assert_transforms_close(
            transform @ group.exp(omega),
            group.exp(transform.adjoint() @ omega) @ transform,
        )

    def test_right_minus_operation(self, group: Type[MatrixLieGroup]):
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a.inverse() @ transform_b
        np.testing.assert_allclose(transform_b.rminus(transform_a), transform_c.log())

    def test_left_minus_operation(self, group: Type[MatrixLieGroup]):
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        np.testing.assert_allclose(transform_a.lminus(transform_b), (transform_a @ transform_b.inverse()).log())

    def test_right_plus_operation(self, group: Type[MatrixLieGroup]):
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a.inverse() @ transform_b
        assert_transforms_close(transform_a.rplus(transform_c.log()), transform_b)

    def test_left_plus_operation(self, group: Type[MatrixLieGroup]):
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a @ transform_b.inverse()
        assert_transforms_close(transform_b.lplus(transform_c.log()), transform_a)

    def test_jacobian_of_log_operation(self, group: Type[MatrixLieGroup]):
        state = group.sample_uniform()
        perturbation = np.random.rand(state.tangent_dim) * 1e-4
        perturbed_state_log = state.plus(perturbation).log()
        linearized_state_log = state.log() + state.jlog() @ perturbation
        np.testing.assert_allclose(perturbed_state_log, linearized_state_log, atol=1e-7)


class TestSpecificGroupOperations(absltest.TestCase):
    """Tests specific to individual groups."""

    def test_so3_rpy_conversion_is_bijective(self):
        transform = SO3.sample_uniform()
        assert_transforms_close(transform, SO3.from_rpy_radians(*transform.as_rpy_radians()))


if __name__ == "__main__":
    absltest.main()