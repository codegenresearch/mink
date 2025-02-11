"""Tests utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_throws_assertion_error_if_shape_invalid(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))  # Invalid shape, should raise AssertionError

    def test_skew_equals_negative(self):
        omega = np.random.randn(3)
        m = utils.skew(omega)
        np.testing.assert_allclose(m.T, -m)

    def test_skew_returns_correct_skew_symmetric_matrix(self):
        omega = np.array([1.0, 2.0, 3.0])
        expected_m = np.array([[0.0, -3.0, 2.0],
                               [3.0, 0.0, -1.0],
                               [-2.0, 1.0, 0.0]])
        np.testing.assert_allclose(utils.skew(omega), expected_m)


if __name__ == "__main__":
    absltest.main()


I have removed the comment at the end of the file to resolve the `SyntaxError`. The test cases are now focused and essential for validating the functionality of the `skew` function. The assertions are concise and directly related to the functionality being tested, and the structure is consistent with the expected format.