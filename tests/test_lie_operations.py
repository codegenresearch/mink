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
        """Verify inverse operation."""
        t = group.sample_uniform()
        assert_transforms_close(t, t.inverse().inverse())

    def test_matrix_conversion(self, group: Type[MatrixLieGroup]):
        """Ensure matrix conversion is bijective."""
        t = group.sample_uniform()
        assert_transforms_close(t, group.from_matrix(t.as_matrix()))

    def test_log_exp(self, group: Type[MatrixLieGroup]):
        """Validate log and exp operations."""
        t = group.sample_uniform()
        tangent = t.log()
        self.assertEqual(tangent.shape, (group.tangent_dim,))
        exp_t = group.exp(tangent)
        assert_transforms_close(t, exp_t)
        np.testing.assert_allclose(tangent, exp_t.log())

    def test_adjoint(self, group: Type[MatrixLieGroup]):
        t = group.sample_uniform()
        omega = np.random.randn(group.tangent_dim)
        assert_transforms_close(
            t @ group.exp(omega),
            group.exp(t.adjoint() @ omega) @ t,
        )

    def test_rminus(self, group: Type[MatrixLieGroup]):
        a = group.sample_uniform()
        b = group.sample_uniform()
        c = a.inverse() @ b
        np.testing.assert_allclose(b.rminus(a), c.log())

    def test_lminus(self, group: Type[MatrixLieGroup]):
        a = group.sample_uniform()
        b = group.sample_uniform()
        np.testing.assert_allclose(a.lminus(b), (a @ b.inverse()).log())

    def test_rplus(self, group: Type[MatrixLieGroup]):
        a = group.sample_uniform()
        b = group.sample_uniform()
        c = a.inverse() @ b
        assert_transforms_close(a.rplus(c.log()), b)

    def test_lplus(self, group: Type[MatrixLieGroup]):
        a = group.sample_uniform()
        b = group.sample_uniform()
        c = a @ b.inverse()
        assert_transforms_close(b.lplus(c.log()), a)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        state = group.sample_uniform()
        w = np.random.rand(state.tangent_dim) * 1e-4
        state_pert = state.plus(w).log()
        state_lin = state.log() + state.jlog() @ w
        np.testing.assert_allclose(state_pert, state_lin, atol=1e-7)


class TestSpecificGroupOps(absltest.TestCase):
    """Tests specific to individual groups."""

    def test_so3_rpy_conversion(self):
        t = SO3.sample_uniform()
        assert_transforms_close(t, SO3.from_rpy_radians(*t.as_rpy_radians()))


if __name__ == "__main__":
    absltest.main()