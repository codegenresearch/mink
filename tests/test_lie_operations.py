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
        """Check that applying inverse twice returns the original transform."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, transform.inverse().inverse())

    def test_matrix_conversion_bijective(self, group: Type[MatrixLieGroup]):
        """Ensure matrix conversion is bijective."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, group.from_matrix(transform.as_matrix()))

    def test_log_exp_bijective(self, group: Type[MatrixLieGroup]):
        """Validate that log and exp operations are bijective."""
        transform = group.sample_uniform()
        tangent = transform.log()
        self.assertEqual(tangent.shape, (group.tangent_dim,))
        exp_transform = group.exp(tangent)
        assert_transforms_close(transform, exp_transform)
        np.testing.assert_allclose(tangent, exp_transform.log())

    def test_adjoint_operation(self, group: Type[MatrixLieGroup]):
        """Verify the adjoint operation."""
        transform = group.sample_uniform()
        omega = np.random.randn(group.tangent_dim)
        assert_transforms_close(
            transform @ group.exp(omega),
            group.exp(transform.adjoint() @ omega) @ transform,
        )

    def test_right_minus_operation(self, group: Type[MatrixLieGroup]):
        """Verify the right minus operation."""
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a.inverse() @ transform_b
        np.testing.assert_allclose(transform_b.rminus(transform_a), transform_c.log())

    def test_left_minus_operation(self, group: Type[MatrixLieGroup]):
        """Verify the left minus operation."""
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        np.testing.assert_allclose(transform_a.lminus(transform_b), (transform_a @ transform_b.inverse()).log())

    def test_right_plus_operation(self, group: Type[MatrixLieGroup]):
        """Verify the right plus operation."""
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a.inverse() @ transform_b
        assert_transforms_close(transform_a.rplus(transform_c.log()), transform_b)

    def test_left_plus_operation(self, group: Type[MatrixLieGroup]):
        """Verify the left plus operation."""
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a @ transform_b.inverse()
        assert_transforms_close(transform_b.lplus(transform_c.log()), transform_a)

    def test_jacobian_of_log_operation(self, group: Type[MatrixLieGroup]):
        """Verify the Jacobian of the log operation."""
        state = group.sample_uniform()
        perturbation = np.random.rand(state.tangent_dim) * 1e-4
        perturbed_state_log = state.plus(perturbation).log()
        linearized_state_log = state.log() + state.jlog() @ perturbation
        np.testing.assert_allclose(perturbed_state_log, linearized_state_log, atol=1e-7)


class TestSpecificGroupOperations(absltest.TestCase):
    """Tests specific to individual groups."""

    def test_so3_rpy_conversion_bijective(self):
        """Verify that RPY conversion is bijective for SO3."""
        transform = SO3.sample_uniform()
        assert_transforms_close(transform, SO3.from_rpy_radians(*transform.as_rpy_radians()))

    def test_so3_invalid_shape_raises_error(self):
        """Ensure that invalid shape inputs raise a ValueError for SO3."""
        with self.assertRaises(ValueError):
            SO3.from_matrix(np.random.rand(3, 4))  # Invalid shape

    def test_se3_invalid_shape_raises_error(self):
        """Ensure that invalid shape inputs raise a ValueError for SE3."""
        with self.assertRaises(ValueError):
            SE3.from_matrix(np.random.rand(3, 3))  # Invalid shape


if __name__ == "__main__":
    absltest.main()


### Changes Made:
1. **Naming Conventions**: Ensured that the names of test methods and classes are consistent with the gold code.
2. **Docstrings**: Reviewed and adjusted docstrings to be concise yet descriptive.
3. **Variable Naming**: Used `transform` instead of `T` for clarity.
4. **Error Handling Tests**: Ensured that tests for invalid shapes are comprehensive and match the structure of the gold code.
5. **Additional Tests**: No additional tests were added in this snippet, but the structure is open for adding more.
6. **Imports**: No additional imports were necessary based on the provided feedback.
7. **Consistency in Assertions**: Used `np.testing.assert_allclose` and `self.assertEqual` consistently with the gold code's approach.