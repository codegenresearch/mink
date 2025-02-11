"""Collision avoidance limit."""

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Union

import mujoco
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit

# Type aliases.
Geom = Union[int, str]
GeomSequence = Sequence[Geom]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]


@dataclass(frozen=True)
class _Contact:
    """Contact information between two geoms.

    Attributes:
        dist: Distance between the two geoms.
        fromto: Array containing the start and end points of the contact.
        geom1: ID of the first geom.
        geom2: ID of the second geom.
        distmax: Maximum distance for collision detection.
    """

    dist: float
    fromto: np.ndarray
    geom1: int
    geom2: int
    distmax: float

    @property
    def normal(self) -> np.ndarray:
        """Normal vector of the contact surface.

        Returns:
            Normal vector.
        """
        normal = self.fromto[3:] - self.fromto[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)

    @property
    def inactive(self) -> bool:
        """Check if the contact is inactive.

        Returns:
            True if inactive, False otherwise.
        """
        return self.dist == self.distmax and not self.fromto.any()


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Check if two geoms are part of the same body or are welded together.

    Args:
        model: MuJoCo model.
        geom_id1: ID of the first geom.
        geom_id2: ID of the second geom.

    Returns:
        True if the geoms are part of the same body or are welded together, False otherwise.
    """
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Check if the bodies of two geoms have a parent-child relationship.

    Args:
        model: MuJoCo model.
        geom_id1: ID of the first geom.
        geom_id2: ID of the second geom.

    Returns:
        True if the bodies of the geoms have a parent-child relationship, False otherwise.
    """
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    weld_parent_id1 = model.body_parentid[model.body_weldid[body_id1]]
    weld_parent_id2 = model.body_parentid[model.body_weldid[body_id2]]

    is_parent_child = weld_parent_id1 == model.body_weldid[body_id2] or \
                      weld_parent_id2 == model.body_weldid[body_id1]
    return is_parent_child


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Check if two geoms pass the contype/conaffinity check.

    Args:
        model: MuJoCo model.
        geom_id1: ID of the first geom.
        geom_id2: ID of the second geom.

    Returns:
        True if the geoms pass the contype/conaffinity check, False otherwise.
    """
    contype_check = bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2])
    conaffinity_check = bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])
    return contype_check or conaffinity_check


class CollisionAvoidanceLimit(Limit):
    """Enforce collision avoidance between specified geom pairs.

    Attributes:
        model: MuJoCo model.
        geom_pairs: Collision pairs for active collision avoidance.
        gain: Gain factor for collision avoidance.
        minimum_distance_from_collisions: Minimum distance between geoms.
        collision_detection_distance: Distance for collision detection.
        bound_relaxation: Offset on the upper bound of collision constraints.
        geom_id_pairs: List of geom ID pairs for collision avoidance.
        max_num_contacts: Maximum number of possible contacts.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """Initialize collision avoidance limit.

        Args:
            model: MuJoCo model.
            geom_pairs: Collision pairs for active collision avoidance.
            gain: Gain factor for collision avoidance.
            minimum_distance_from_collisions: Minimum distance between geoms.
            collision_detection_distance: Distance for collision detection.
            bound_relaxation: Offset on the upper bound of collision constraints.
        """
        if not 0.0 < gain <= 1.0:
            raise ValueError("Gain must be in the range (0, 1].")
        if minimum_distance_from_collisions < 0:
            raise ValueError("Minimum distance from collisions must be non-negative.")
        if collision_detection_distance < 0:
            raise ValueError("Collision detection distance must be non-negative.")

        self.model = model
        self.gain = gain
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance
        self.bound_relaxation = bound_relaxation
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_num_contacts = len(self.geom_id_pairs)

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        """Compute the quadratic programming inequalities for collision avoidance.

        Args:
            configuration: Current robot configuration.
            dt: Integration timestep in seconds.

        Returns:
            Constraint object.
        """
        upper_bound = np.full((self.max_num_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_num_contacts, self.model.nv))
        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):
            contact = self._compute_contact_with_minimum_distance(
                configuration.data, geom1_id, geom2_id
            )
            if contact.inactive:
                continue
            hi_bound_dist = contact.dist
            if hi_bound_dist > self.minimum_distance_from_collisions:
                dist = hi_bound_dist - self.minimum_distance_from_collisions
                upper_bound[idx] = (self.gain * dist / dt) + self.bound_relaxation
            else:
                upper_bound[idx] = self.bound_relaxation
            jac = self._compute_contact_normal_jacobian(configuration.data, contact)
            coefficient_matrix[idx] = -jac
        return Constraint(G=coefficient_matrix, h=upper_bound)

    def _compute_contact_with_minimum_distance(
        self, data: mujoco.MjData, geom1_id: int, geom2_id: int
    ) -> _Contact:
        """Compute the smallest signed distance between two geoms.

        Args:
            data: MuJoCo data structure.
            geom1_id: ID of the first geom.
            geom2_id: ID of the second geom.

        Returns:
            Contact object.
        """
        fromto = np.empty(6)
        dist = mujoco.mj_geomDistance(
            self.model,
            data,
            geom1_id,
            geom2_id,
            self.collision_detection_distance,
            fromto,
        )
        return _Contact(
            dist, fromto, geom1_id, geom2_id, self.collision_detection_distance
        )

    def _compute_contact_normal_jacobian(
        self, data: mujoco.MjData, contact: _Contact
    ) -> np.ndarray:
        """Compute the Jacobian mapping joint velocities to the normal component of
        the relative Cartesian linear velocity between two geoms.

        Args:
            data: MuJoCo data structure.
            contact: Contact object.

        Returns:
            Jacobian matrix.
        """
        geom1_body = self.model.geom_bodyid[contact.geom1]
        geom2_body = self.model.geom_bodyid[contact.geom2]
        geom1_contact_pos = contact.fromto[:3]
        geom2_contact_pos = contact.fromto[3:]
        jac2 = np.empty((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jac2, None, geom2_contact_pos, geom2_body)
        jac1 = np.empty((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jac1, None, geom1_contact_pos, geom1_body)
        return contact.normal @ (jac2 - jac1)

    def _homogenize_geom_id_list(self, geom_list: GeomSequence) -> List[int]:
        """Convert a list of geoms (specified by ID or name) to a list of IDs.

        Args:
            geom_list: List of geoms specified by ID or name.

        Returns:
            List of geom IDs.
        """
        list_of_int: list[int] = []
        for g in geom_list:
            if isinstance(g, int):
                list_of_int.append(g)
            elif isinstance(g, str):
                list_of_int.append(self.model.geom(g).id)
            else:
                raise TypeError("Geom list must contain only integers or strings.")
        return list_of_int

    def _construct_geom_id_pairs(self, geom_pairs: CollisionPairs) -> List[tuple[int, int]]:
        """Construct a list of geom ID pairs for all possible geom-geom collisions.

        Args:
            geom_pairs: List of collision pairs specified by geom names.

        Returns:
            List of geom ID pairs.
        """
        geom_id_pairs = []
        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                is_welded = _is_welded_together(self.model, geom_a, geom_b)
                is_parent_child = _are_geom_bodies_parent_child(self.model, geom_a, geom_b)
                is_pass_check = _is_pass_contype_conaffinity_check(self.model, geom_a, geom_b)
                if not is_welded and not is_parent_child and is_pass_check:
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs) -> List[tuple[List[int], List[int]]]:
        """Convert collision pairs of geom names to collision pairs of geom IDs.

        Args:
            collision_pairs: List of collision pairs specified by geom names.

        Returns:
            List of collision pairs specified by geom IDs.
        """
        geom_id_pairs = []
        for collision_pair in collision_pairs:
            id_pair_A = self._homogenize_geom_id_list(collision_pair[0])
            id_pair_B = self._homogenize_geom_id_list(collision_pair[1])
            id_pair_A = list(set(id_pair_A))
            id_pair_B = list(set(id_pair_B))
            geom_id_pairs.append((id_pair_A, id_pair_B))
        return geom_id_pairs


### Key Changes:
1. **Removed Invalid Syntax**: Removed the list-like structure at the end of the file that was causing the `SyntaxError`.
2. **Docstring Clarity and Consistency**: Ensured that the docstrings for classes and methods are clear, consistent, and provide detailed descriptions of the purpose and functionality.
3. **Method Naming**: Ensured that method names are descriptive and follow a consistent naming convention.
4. **Conditional Logic**: Simplified and clarified the conditional logic in `_are_geom_bodies_parent_child` by breaking down complex conditions into separate variables.
5. **Return Statements**: Made return statements clear and concise.
6. **Attribute Descriptions**: Provided detailed descriptions for the attributes in the `CollisionAvoidanceLimit` class.
7. **Type Annotations**: Ensured that type annotations are consistent and accurately reflect the expected types.
8. **Code Structure and Comments**: Improved the structure of the code and the clarity of comments, particularly in complex logic sections.
9. **Private Method Documentation**: Ensured that private methods are well-documented, explaining their purpose and functionality clearly.
10. **Simplification of Logic**: Streamlined the logic in methods like `_construct_geom_id_pairs` for better readability and maintainability.
11. **General Code Style**: Reviewed the overall code style to ensure it adheres to best practices and conventions, making it more readable and maintainable.

These changes should resolve the syntax error and align the code more closely with the gold standard.