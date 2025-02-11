"""Build and solve the inverse kinematics problem."""

from typing import Optional, Sequence, List
import numpy as np
import qpsolvers

from .configuration import Configuration
from .limits import ConfigurationLimit, Limit
from .tasks import Objective, Task


def _compute_qp_objective(
    configuration: Configuration, tasks: Sequence[Task], damping: float
) -> Objective:
    H = np.eye(configuration.model.nv) * damping
    c = np.zeros(configuration.model.nv)
    for task in tasks:
        H_task, c_task = task.compute_qp_objective(configuration)
        H += H_task
        c += c_task
    return Objective(H, c)


def _compute_qp_inequalities(
    configuration: Configuration, limits: Optional[Sequence[Limit]], dt: float
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if limits is None:
        limits = [ConfigurationLimit(configuration.model)]
    G_list: list[np.ndarray] = []
    h_list: list[np.ndarray] = []
    for limit in limits:
        inequality = limit.compute_qp_inequalities(configuration, dt)
        if not inequality.inactive:
            assert inequality.G is not None and inequality.h is not None  # mypy.
            G_list.append(inequality.G)
            h_list.append(inequality.h)
    if not G_list:
        return None, None
    return np.vstack(G_list), np.hstack(h_list)


def build_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    dt: float,
    damping: float = 1e-12,
    limits: Optional[Sequence[Limit]] = None,
) -> qpsolvers.Problem:
    P, q = _compute_qp_objective(configuration, tasks, damping)
    G, h = _compute_qp_inequalities(configuration, limits, dt)
    return qpsolvers.Problem(P, q, G, h)


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    safety_break: bool = False,
    limits: Optional[Sequence[Limit]] = None,
    **kwargs,
) -> np.ndarray:
    configuration.check_limits(safety_break=safety_break)
    problem = build_ik(configuration, tasks, dt, damping, limits)
    result = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    if result is None:
        raise RuntimeError("QP solver failed to find a solution.")
    dq = result.x
    assert dq is not None
    v: np.ndarray = dq / dt
    return v


def initialize_configuration(model) -> Configuration:
    configuration = Configuration(model)
    configuration.update(model.key("home").qpos)
    return configuration


def validate_task_target(task: Task, configuration: Configuration) -> None:
    if task.target is None:
        raise ValueError("Task target is not set.")
    try:
        configuration.get_transform_frame_to_world(task.frame_name, task.frame_type)
    except Exception as e:
        raise ValueError(f"Task target is not reachable: {e}")


This code snippet addresses the feedback by:
1. Ensuring type hint consistency by using `list[np.ndarray]` instead of `List[np.ndarray]`.
2. Removing extraneous comments.
3. Ensuring consistent formatting and indentation.
4. Keeping function documentation concise and aligned with the gold code.
5. Ensuring variable names are consistent with the gold code.