"""Tests utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_throws_assertion_error_if_shape_invalid(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))

    def test_skew_equals_negative(self):
        m = utils.skew(np.random.randn(3))
        np.testing.assert_allclose(m.T, -m)


if __name__ == "__main__":
    absltest.main()


I have removed the inline comment that was causing the `SyntaxError` by ensuring it starts with a `#`. This should resolve the syntax error and allow the tests to run successfully.