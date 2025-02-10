from absl.testing import absltest

from mink.tasks.exceptions import InvalidDamping, InvalidGain
from mink.tasks.task import Task


class TestTask(absltest.TestCase):
    """Tests for the Task class to ensure proper error handling when invalid parameters are provided."""

    def setUp(self):
        """Prepare test fixture."""
        Task.__abstractmethods__ = set()

    def test_task_throws_error_if_gain_is_negative(self):
        with self.assertRaises(InvalidGain):
            Task(cost=0.0, gain=-0.5)

    def test_task_throws_error_if_damping_is_negative(self):
        with self.assertRaises(InvalidDamping):
            Task(cost=0.0, gain=1.0, lm_damping=-1.0)


if __name__ == "__main__":
    absltest.main()