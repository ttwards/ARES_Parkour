"""Joint torque monitoring manager for visualization."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv


class JointTorqueTerm:
    """Individual joint torque term."""

    def __init__(self, joint_idx: int, joint_name: str, monitor: JointTorqueMonitor):
        self.joint_idx = joint_idx
        self.joint_name = joint_name
        self._monitor = monitor

    def __call__(self, env_idx: int = 0) -> torch.Tensor:
        """Get torque value for this joint."""
        return self._monitor._torque_buf[env_idx, self.joint_idx:self.joint_idx + 1]


class JointTorqueMonitor:
    """Manager for monitoring and storing joint torque data.

    This class mimics the structure of other managers to work with ManagerLiveVisualizer.
    """

    def __init__(self, env: ParkourManagerBasedRLEnv):
        """Initialize the joint torque monitor.

        Args:
            env: The environment instance.
        """
        self._env = env
        self._num_envs = env.num_envs
        self._device = env.device

        # Get robot asset
        self._robot = env.scene["robot"]
        self._num_joints = self._robot.num_joints

        # Storage for torque data
        self._torque_buf = torch.zeros(self._num_envs, self._num_joints, device=self._device)
        self._power_buf = torch.zeros(self._num_envs, self._num_joints, device=self._device)

        # Create terms for each joint (required by ManagerLiveVisualizer)
        self._terms = {}
        self._term_names = []
        for i, joint_name in enumerate(self._robot.joint_names):
            term = JointTorqueTerm(i, joint_name, self)
            self._terms[joint_name] = term
            self._term_names.append(joint_name)

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs

    @property
    def device(self) -> torch.device:
        """Device on which the tensors are stored."""
        return self._device

    @property
    def active_terms(self) -> list[str]:
        """List of active terms (joint names)."""
        return self._term_names

    def get_active_iterable_terms(self, env_idx: int = 0):
        """Get active terms data for visualization.

        This method is required by ManagerLiveVisualizer.

        Args:
            env_idx: Environment index to get data for.

        Returns:
            Iterable of (name, term_data) tuples.
        """
        # Return all joint torques as a single list for plotting on one graph
        all_torques = self._torque_buf[env_idx].cpu().numpy().tolist()
        yield ("All Joints", all_torques)

    def get_term(self, name: str) -> torch.Tensor:
        """Get torque data for a specific joint.

        Args:
            name: Joint name

        Returns:
            Torque tensor of shape (num_envs,)
        """
        if name in self._term_names:
            idx = self._term_names.index(name)
            return self._torque_buf[:, idx]
        else:
            raise KeyError(f"Joint '{name}' not found. Available joints: {self._term_names}")

    def compute(self) -> torch.Tensor:
        """Update torque and power buffers.

        Returns:
            Torque buffer of shape (num_envs, num_joints)
        """
        # Get current joint velocities and applied torques
        joint_vel = self._robot.data.joint_vel  # (num_envs, num_joints)
        applied_torque = self._robot.data.applied_torque  # (num_envs, num_joints)

        # Update buffers
        self._torque_buf[:] = applied_torque
        self._power_buf[:] = torch.abs(joint_vel * applied_torque)

        return self._torque_buf

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset buffers for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            self._torque_buf.zero_()
            self._power_buf.zero_()
        else:
            self._torque_buf[env_ids] = 0.0
            self._power_buf[env_ids] = 0.0

    def __str__(self) -> str:
        """String representation."""
        msg = f"<JointTorqueMonitor> contains {len(self._term_names)} joints.\n"
        msg += f"Names: {self._term_names}"
        return msg
