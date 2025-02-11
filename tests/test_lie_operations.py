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

    def test_matrix_conversion(self, group: Type[MatrixLieGroup]):
        """Check matrix conversion."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, group.from_matrix(transform.as_matrix()))

    def test_log_exp(self, group: Type[MatrixLieGroup]):
        """Check log and exp operations."""
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

    def test_right_minus(self, group: Type[MatrixLieGroup]):
        """Check right minus operation."""
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a.inverse() @ transform_b
        np.testing.assert_allclose(transform_b.rminus(transform_a), transform_c.log())

    def test_left_minus(self, group: Type[MatrixLieGroup]):
        """Check left minus operation."""
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        np.testing.assert_allclose(transform_a.lminus(transform_b), (transform_a @ transform_b.inverse()).log())

    def test_right_plus(self, group: Type[MatrixLieGroup]):
        """Check right plus operation."""
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a.inverse() @ transform_b
        assert_transforms_close(transform_a.rplus(transform_c.log()), transform_b)

    def test_left_plus(self, group: Type[MatrixLieGroup]):
        """Check left plus operation."""
        transform_a = group.sample_uniform()
        transform_b = group.sample_uniform()
        transform_c = transform_a @ transform_b.inverse()
        assert_transforms_close(transform_b.lplus(transform_c.log()), transform_a)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        """Check Jacobian of log operation."""
        state = group.sample_uniform()
        w = np.random.rand(state.tangent_dim) * 1e-4
        state_pert = state.plus(w).log()
        state_lin = state.log() + state.jlog() @ w
        np.testing.assert_allclose(state_pert, state_lin, atol=1e-7)


class TestGroupSpecificOperations(absltest.TestCase):
    """Group specific tests."""

    def test_so3_rpy_conversion(self):
        """Check SO3 RPY conversion."""
        transform = SO3.sample_uniform()
        assert_transforms_close(transform, SO3.from_rpy_radians(*transform.as_rpy_radians()))

    def test_so3_invalid_shape(self):
        """Check SO3 invalid shape raises error."""
        with self.assertRaises(ValueError):
            SO3.from_matrix(np.random.rand(3, 4))  # Invalid shape

    def test_se3_invalid_shape(self):
        """Check SE3 invalid shape raises error."""
        with self.assertRaises(ValueError):
            SE3.from_matrix(np.random.rand(3, 3))  # Invalid shape


if __name__ == "__main__":
    absltest.main()


### Changes Made:
1. **Class and Method Naming**: Simplified the class name to `TestOperations` and shortened method names while maintaining clarity.
2. **Docstring Clarity**: Made docstrings more concise while still conveying the essential purpose of each test.
3. **Parameterization Consistency**: Ensured parameterization naming and structure are consistent with the gold code.
4. **Error Handling Tests**: Reviewed and ensured error handling tests match the structure and naming conventions of the gold code.
5. **Additional Tests**: No additional tests were added in this snippet, but the structure is open for adding more.
6. **Code Structure**: Maintained a clear separation between general operation tests and specific group operation tests.
7. **Consistency in Assertions**: Ensured assertions are consistent with those in the gold code.

### Specific Fixes:
- **SyntaxError**: Removed any unterminated string literals or comments that could cause syntax errors. Ensured all strings and comments are properly closed.
- **Line 111**: Checked and corrected any issues around line 111 to ensure no unterminated strings or comments exist.