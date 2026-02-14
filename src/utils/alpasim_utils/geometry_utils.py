import numbers
import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List
from enum import IntEnum

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2

class SE2Index(IntEnum):
    """Index mapping for state se2 (x,y,θ) arrays."""

    X = 0
    Y = 1
    HEADING = 2

def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

def convert_absolute_to_relative_se2_array(
    origin: StateSE2, state_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Converts an StateSE2 array from global to relative coordinates.
    :param origin: origin pose of relative coords system
    :param state_se2_array: array of SE2 states with (x,y,θ) in last dim
    :return: SE2 coords array in relative coordinates
    """
    assert len(SE2Index) == state_se2_array.shape[-1]

    theta = -origin.heading
    origin_array = np.array([[origin.x, origin.y, origin.heading]], dtype=np.float64)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points_rel = state_se2_array - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[:, 2] = normalize_angle(points_rel[:, 2])

    return points_rel

@dataclass(frozen=True)
class Sim2:
    """Similarity(2) lie group object.

    Args:
        R: array of shape (2x2) representing 2d rotation matrix
        t: array of shape (2,) representing 2d translation
        s: scaling factor
    """

    R: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    s: float

    def __post_init__(self) -> None:
        """Check validity of rotation, translation, and scale arguments.

        Raises:
            ValueError: If `R` is not shape (3,3), `t` is not shape (2,),
                 or `s` is not a numeric type.
            ZeroDivisionError: If `s` is close or equal to zero.
        """
        assert self.R.shape == (2, 2)
        assert self.t.shape == (2,)

        if not isinstance(self.s, numbers.Number):
            raise ValueError("Scale `s` must be a numeric type!")
        if math.isclose(self.s, 0.0):
            raise ZeroDivisionError(
                "3x3 matrix formation would require division by zero"
            )
    
    def __repr__(self) -> str:
        """Return a human-readable string representation of the class."""
        trans = np.round(self.t, 2)
        return (
            f"Angle (deg.): {self.theta_deg:.1f}, Trans.: {trans}, Scale: {self.s:.1f}"
        )

    def __eq__(self, other: object) -> bool:
        """Check for equality with other Sim(2) object."""
        if not isinstance(other, Sim2):
            return False

        if not np.isclose(self.scale, other.scale):
            return False

        if not np.allclose(self.rotation, other.rotation):
            return False

        if not np.allclose(self.translation, other.translation):
            return False

        return True

    @property
    def theta_deg(self) -> float:
        """Recover the rotation angle `theta` (in degrees) from the 2d rotation matrix.

        Note: the first column of the rotation matrix R provides sine and cosine of theta,
            since R is encoded as [c,-s]
                                  [s, c]

        We use the following identity: tan(theta) = s/c = (opp/hyp) / (adj/hyp) = opp/adj

        Returns:
            Rotation angle from the 2d rotation matrix.
        """
        c, s = self.R[0, 0], self.R[1, 0]
        theta_rad = np.arctan2(s, c)
        return float(np.rad2deg(theta_rad))

    @property
    def rotation(self) -> npt.NDArray[np.float64]:
        """Return the 2x2 rotation matrix."""
        return self.R

    @property
    def translation(self) -> npt.NDArray[np.float64]:
        """Return the (2,) translation vector."""
        return self.t

    @property
    def scale(self) -> float:
        """Return the scale."""
        return self.s

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Calculate 3*3 matrix group equivalent."""
        T = np.zeros((3, 3))
        T[:2, :2] = self.R
        T[:2, 2] = self.t
        T[2, 2] = 1 / self.s
        return T

    def compose(self, S):
        """Composition with another Sim2.

        This can be understood via block matrix multiplication, if self is parameterized as (R1,t1,s1)
        and if `S` is parameterized as (R2,t2,s2):

        [R1  t1]   [R2  t2]   [R1 @ R2   R1@t2 + t1/s2]
        [0 1/s1] @ [0 1/s2] = [ 0          1/(s1*s2)  ]

        Args:
            S: Similarity(2) transformation.

        Returns:
            Composed Similarity(2) transformation.
        """
        # fmt: off
        return Sim2(
            R=self.R @ S.R,
            t=self.R @ S.t + ((1.0 / S.s) * self.t), 
            s=self.s * S.s
        )
        # fmt: on

    def inverse(self):
        """Return the inverse."""
        Rt = self.R.T
        sRt = -Rt @ (self.s * self.t)
        return Sim2(Rt, sRt, 1.0 / self.s)

    def transform_from(self, points_xy):
        """Transform point cloud from reference frame a to b.

        If the points are in frame A, and our Sim(3) transform is defines as bSa, then we get points
        back in frame B:
            p_b = bSa * p_a
        Action on a point p is s*(R*p+t).

        Args:
            points_xy: Nx2 array representing 2d points in frame A.

        Returns:
            Nx2 array representing 2d points in frame B.

        Raises:
            ValueError: if `points_xy` isn't in R^2.
        """
        if not points_xy.ndim == 2:
            raise ValueError("Input points are not 2-dimensional.")
        assert points_xy.shape[1] == 2
        # (2,2) x (2,N) + (2,1) = (2,N) -> transpose
        transformed_point_cloud = (points_xy @ self.R.T) + self.t

        # now scale points
        return transformed_point_cloud * self.s

    @classmethod
    def from_json(cls, json_fpath: Path):
        """Generate class inst. from a JSON file containing Sim(2) parameters as flattened matrices (row-major)."""
        with json_fpath.open("r") as f:
            json_data = json.load(f)

        R = np.array(json_data["R"]).reshape(2, 2)
        t = np.array(json_data["t"]).reshape(2)
        s = float(json_data["s"])
        return cls(R, t, s)

    @classmethod
    def from_matrix(cls, T: npt.NDArray[np.float64]):
        """Generate class instance from a 3x3 Numpy matrix."""
        if np.isclose(T[2, 2], 0.0):
            raise ZeroDivisionError(
                "Sim(2) scale calculation would lead to division by zero."
            )

        R = T[:2, :2]
        t = T[:2, 2]
        s = 1 / T[2, 2]
        return cls(R, t, s)
