from absl.testing import absltest

from mink.tasks.exceptions import InvalidDamping, InvalidGain
from mink.tasks.task import Task


class TaskTest(absltest.TestCase):
    """Test the Task abstract base class."""

    def setUp(self):
        """Set up test fixtures."""
        Task.__abstractmethods__ = set()

    def test_task_raises_invalid_gain_error_when_gain_is_negative(self):
        """Test that Task raises InvalidGain when gain is negative."""
        with self.assertRaises(InvalidGain):
            Task(cost=0.0, gain=-0.5)

    def test_task_raises_invalid_damping_error_when_lm_damping_is_negative(self):
        """Test that Task raises InvalidDamping when lm_damping is negative."""
        with self.assertRaises(InvalidDamping):
            Task(cost=0.0, gain=1.0, lm_damping=-1.0)


if __name__ == "__main__":
    absltest.main()