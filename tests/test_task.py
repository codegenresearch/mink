from absl.testing import absltest

from mink.tasks.exceptions import InvalidDamping, InvalidGain
from mink.tasks.task import Task


class TestTask(absltest.TestCase):
    """Test the Task class."""

    def setUp(self):
        """Prepare test fixture."""
        Task.__abstractmethods__ = set()

    def test_task_raises_error_if_gain_is_negative(self):
        with self.assertRaises(InvalidGain):
            Task(cost=0.0, gain=-0.5)

    def test_task_raises_error_if_lm_damping_is_negative(self):
        with self.assertRaises(InvalidDamping):
            Task(cost=0.0, gain=1.0, lm_damping=-1.0)


if __name__ == "__main__":
    absltest.main()