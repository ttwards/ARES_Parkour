from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from .joint_actions import DelayedJointPositionAction, DelayedJointVelocityAction


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = DelayedJointPositionAction
    delay_update_global_steps: int = 24 * 8000
    history_length: int = 8
    action_delay_steps: list[int] | int = [1, 1]
    use_delay: bool = False


@configclass
class DelayedJointVelocityActionCfg(JointVelocityActionCfg):
    class_type: type[ActionTerm] = DelayedJointVelocityAction
    delay_update_global_steps: int = 24 * 8000
    history_length: int = 8
    action_delay_steps: list[int] | int = [1, 1]
    use_delay: bool = False
