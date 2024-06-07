from builtins import Exception
from typing import Dict, Optional, List, Tuple

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from controller_manager_msgs.srv import ListControllers, ListControllersResponse, SwitchController, \
    SwitchControllerResponse
from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped, Vector3Stamped, Vector3
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import os
if 'GITHUB_WORKFLOW' not in os.environ:
    from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortGoal
    from tmc_manipulation_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

from giskard_msgs.msg import MoveResult, CollisionEntry, MoveGoal, WorldResult
from giskard_msgs.srv import DyeGroupResponse, GetGroupInfoResponse
from giskardpy.data_types import goal_parameter
from giskardpy.goals.realtime_goals import RealTimePointingPose
from giskardpy.goals.suturo import Reaching, Placing, Retracting, Tilting, TakePose, OpenDoorGoal, MoveAroundDoor
from giskardpy.python_interface.python_interface import GiskardWrapper
from giskardpy.suturo_types import GripperTypes
from giskardpy.tasks.task import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.tree.control_modes import ControlModes


class OldGiskardWrapper(GiskardWrapper):

    def __init__(self, node_name: str = 'giskard', check_controller: bool = True):

        super().__init__(node_name, avoid_name_conflict=True)

        if check_controller and self.world.get_control_mode() == ControlModes.close_loop:
            # TODO: create config for robots
            start_con = ['realtime_body_controller_real']
            stop_con = ['arm_trajectory_controller', 'head_trajectory_controller']
            switch_controllers_srv_name = '/hsrb/controller_manager/switch_controller'
            list_controllers_srv_name = '/hsrb/controller_manager/list_controllers'

            self.list_controller_srv = rospy.ServiceProxy(name=list_controllers_srv_name,
                                                          service_class=ListControllers)

            self.set_closed_loop_controllers(stop=stop_con,
                                             start=start_con,
                                             switch_controller_srv=switch_controllers_srv_name)

            if not self.check_controllers_active(
                    stopped_controllers=stop_con,
                    running_controllers=start_con):
                raise Exception(f'Controllers are configured incorrectly. Look at rqt_controller_manager.')

    def execute(self, wait: bool = True, add_default: bool = True) -> MoveResult:
        if add_default:
            self.add_default_end_motion_conditions()
        return super().execute(wait)

    def projection(self, wait: bool = True) -> MoveResult:
        self.add_default_end_motion_conditions()
        return super().projection(wait)

    def _create_action_goal(self) -> MoveGoal:
        if not self.motion_goals._collision_entries:
            self.motion_goals.avoid_all_collisions()
        action_goal = MoveGoal()
        action_goal.monitors = self.monitors.get_monitors()
        action_goal.goals = self.motion_goals.get_goals()
        self.clear_motion_goals_and_monitors()
        return action_goal

    # %% predefined goals
    def set_joint_goal(self,
                       goal_state: Dict[str, float],
                       group_name: Optional[str] = None,
                       weight: Optional[float] = None,
                       max_velocity: Optional[float] = None,
                       add_monitor: bool = True,
                       **kwargs: goal_parameter):
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param add_monitor: if True, adds a monitor as end_condition to check if the goal was reached.
        :param max_velocity: will be applied to all joints
        """
        if add_monitor:
            end_condition = self.monitors.add_joint_position(goal_state=goal_state)
        else:
            end_condition = ''
        self.motion_goals.add_joint_position(goal_state=goal_state,
                                             group_name=group_name,
                                             weight=weight,
                                             max_velocity=max_velocity,
                                             end_condition=end_condition,
                                             **kwargs)

    def set_cart_goal(self,
                      goal_pose: PoseStamped,
                      tip_link: str,
                      root_link: str,
                      tip_group: Optional[str] = None,
                      root_group: Optional[str] = None,
                      reference_linear_velocity: Optional[float] = None,
                      reference_angular_velocity: Optional[float] = None,
                      weight: Optional[float] = None,
                      add_monitor: bool = True,
                      **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param add_monitor: if True, adds a monitor as end_condition to check if the goal was reached.
        :param weight: default WEIGHT_ABOVE_CA
        """
        if add_monitor:
            end_condition = self.monitors.add_cartesian_pose(root_link=root_link,
                                                             root_group=root_group,
                                                             tip_link=tip_link,
                                                             tip_group=tip_group,
                                                             goal_pose=goal_pose)
        else:
            end_condition = ''
        self.motion_goals.add_cartesian_pose(goal_pose=goal_pose,
                                             tip_link=tip_link,
                                             root_link=root_link,
                                             root_group=root_group,
                                             tip_group=tip_group,
                                             reference_linear_velocity=reference_linear_velocity,
                                             reference_angular_velocity=reference_angular_velocity,
                                             weight=weight,
                                             end_condition=end_condition,
                                             **kwargs)

    def set_diff_drive_base_goal(self,
                                 goal_pose: PoseStamped,
                                 tip_link: str,
                                 root_link: str,
                                 tip_group: Optional[str] = None,
                                 root_group: Optional[str] = None,
                                 reference_linear_velocity: Optional[float] = None,
                                 reference_angular_velocity: Optional[float] = None,
                                 weight: Optional[float] = None,
                                 add_monitor: bool = True,
                                 **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param add_monitor: if True, adds a monitor as end_condition to check if the goal was reached.
        """
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} pose reached'
            end_condition = self.monitors.add_cartesian_pose(name=monitor_name,
                                                             root_link=root_link,
                                                             root_group=root_group,
                                                             tip_link=tip_link,
                                                             tip_group=tip_group,
                                                             position_threshold=0.02,
                                                             goal_pose=goal_pose)
        else:
            end_condition = ''
        self.motion_goals.add_diff_drive_base(end_condition=end_condition,
                                              goal_pose=goal_pose,
                                              tip_link=tip_link,
                                              root_link=root_link,
                                              root_group=root_group,
                                              tip_group=tip_group,
                                              reference_linear_velocity=reference_linear_velocity,
                                              reference_angular_velocity=reference_angular_velocity,
                                              weight=weight,
                                              **kwargs)

    def set_straight_cart_goal(self,
                               goal_pose: PoseStamped,
                               tip_link: str,
                               root_link: str,
                               tip_group: Optional[str] = None,
                               root_group: Optional[str] = None,
                               reference_linear_velocity: Optional[float] = None,
                               reference_angular_velocity: Optional[float] = None,
                               weight: Optional[float] = None,
                               add_monitor: bool = True,
                               **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        In contrast to set_cart_goal, this tries to move the tip_link in a straight line to the goal_point.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param add_monitor: if True, adds a monitor as end_condition to check if the goal was reached.
        """
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} pose reached'
            end_condition = self.monitors.add_cartesian_pose(name=monitor_name,
                                                             root_link=root_link,
                                                             root_group=root_group,
                                                             tip_link=tip_link,
                                                             tip_group=tip_group,
                                                             goal_pose=goal_pose)
        else:
            end_condition = ''
        self.motion_goals.add_cartesian_pose_straight(end_condition=end_condition,
                                                      goal_pose=goal_pose,
                                                      tip_link=tip_link,
                                                      tip_group=tip_group,
                                                      root_link=root_link,
                                                      root_group=root_group,
                                                      weight=weight,
                                                      reference_linear_velocity=reference_linear_velocity,
                                                      reference_angular_velocity=reference_angular_velocity,
                                                      **kwargs)

    def set_translation_goal(self,
                             goal_point: PointStamped,
                             tip_link: str,
                             root_link: str,
                             tip_group: Optional[str] = None,
                             root_group: Optional[str] = None,
                             reference_velocity: Optional[float] = 0.2,
                             weight: float = WEIGHT_ABOVE_CA,
                             add_monitor: bool = True,
                             **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_point.
        :param goal_point:
        :param tip_link: tip link of the kinematic chain
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param reference_velocity: m/s
        :param weight:
        :param add_monitor: if True, adds a monitor as end_condition to check if the goal was reached.
        """
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} position reached'
            end_condition = self.monitors.add_cartesian_position(name=monitor_name,
                                                                 root_link=root_link,
                                                                 root_group=root_group,
                                                                 tip_link=tip_link,
                                                                 tip_group=tip_group,
                                                                 goal_point=goal_point)
        else:
            end_condition = ''
        self.motion_goals.add_cartesian_position(end_condition=end_condition,
                                                 goal_point=goal_point,
                                                 tip_link=tip_link,
                                                 root_link=root_link,
                                                 tip_group=tip_group,
                                                 root_group=root_group,
                                                 reference_velocity=reference_velocity,
                                                 weight=weight,
                                                 **kwargs)

    def set_seed_configuration(self, seed_configuration, group_name: Optional[str] = None):
        self.motion_goals.set_seed_configuration(seed_configuration=seed_configuration,
                                                 group_name=group_name)

    def set_straight_translation_goal(self,
                                      goal_point: PointStamped,
                                      tip_link: str,
                                      root_link: str,
                                      tip_group: Optional[str] = None,
                                      root_group: Optional[str] = None,
                                      reference_velocity: float = None,
                                      weight: float = WEIGHT_ABOVE_CA,
                                      add_monitor: bool = True,
                                      **kwargs: goal_parameter):
        """
        Same as set_translation_goal, but will try to move in a straight line.
        """
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} position reached'
            end_condition = self.monitors.add_cartesian_position(name=monitor_name,
                                                                 root_link=root_link,
                                                                 root_group=root_group,
                                                                 tip_link=tip_link,
                                                                 tip_group=tip_group,
                                                                 goal_point=goal_point)
        else:
            end_condition = ''
        self.motion_goals.add_cartesian_position_straight(end_condition=end_condition,
                                                          goal_point=goal_point,
                                                          tip_link=tip_link,
                                                          root_link=root_link,
                                                          tip_group=tip_group,
                                                          root_group=root_group,
                                                          reference_velocity=reference_velocity,
                                                          weight=weight,
                                                          **kwargs)

    def set_rotation_goal(self,
                          goal_orientation: QuaternionStamped,
                          tip_link: str,
                          root_link: str,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          reference_velocity: Optional[float] = None,
                          weight=WEIGHT_ABOVE_CA,
                          add_monitor: bool = True,
                          **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_orientation.
        :param goal_orientation:
        :param tip_link: tip link of kinematic chain
        :param root_link: root link of kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param reference_velocity: rad/s, approx limit
        :param weight:
        :param add_monitor: if True, adds a monitor as end_condition to check if the goal was reached.
        """
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} orientation reached'
            end_condition = self.monitors.add_cartesian_orientation(name=monitor_name,
                                                                    root_link=root_link,
                                                                    root_group=root_group,
                                                                    tip_link=tip_link,
                                                                    tip_group=tip_group,
                                                                    goal_orientation=goal_orientation)
        else:
            end_condition = ''
        self.motion_goals.add_cartesian_orientation(end_condition=end_condition,
                                                    goal_orientation=goal_orientation,
                                                    tip_link=tip_link,
                                                    root_link=root_link,
                                                    tip_group=tip_group,
                                                    root_group=root_group,
                                                    reference_velocity=reference_velocity,
                                                    weight=weight,
                                                    **kwargs)

    def set_align_planes_goal(self,
                              goal_normal: Vector3Stamped,
                              tip_link: str,
                              tip_normal: Vector3Stamped,
                              root_link: str,
                              tip_group: str = None,
                              root_group: str = None,
                              max_angular_velocity: Optional[float] = None,
                              weight: Optional[float] = None,
                              add_monitor: bool = True,
                              **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between tip and root to align tip_normal with goal_normal.
        :param goal_normal:
        :param tip_link: tip link of the kinematic chain
        :param tip_normal:
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip_link is not unique, search in this group for matches.
        :param root_group: if root_link is not unique, search in this group for matches.
        :param max_angular_velocity: rad/s
        :param weight:
        :param add_monitor: if True, adds a monitor as end_condition to check if the goal was reached.
        """
        if add_monitor:
            end_condition = self.monitors.add_vectors_aligned(root_link=root_link,
                                                              tip_link=tip_link,
                                                              goal_normal=goal_normal,
                                                              tip_normal=tip_normal,
                                                              root_group=root_group,
                                                              tip_group=tip_group)
        else:
            end_condition = ''
        self.motion_goals.add_align_planes(end_condition=end_condition,
                                           tip_link=tip_link,
                                           tip_group=tip_group,
                                           tip_normal=tip_normal,
                                           root_link=root_link,
                                           root_group=root_group,
                                           goal_normal=goal_normal,
                                           reference_angular_velocity=max_angular_velocity,
                                           weight=weight,
                                           **kwargs)

    def set_prediction_horizon(self, prediction_horizon: int, **kwargs: goal_parameter):
        """
        Will overwrite the prediction horizon for a single goal.
        Setting it to 1 will turn of acceleration and jerk limits.
        :param prediction_horizon: size of the prediction horizon, a number that should be 1 or above 5.
        """
        self.motion_goals.set_prediction_horizon(prediction_horizon=prediction_horizon,
                                                 **kwargs)

    def set_max_traj_length(self, new_length: float, **kwargs: goal_parameter):
        """
        Overwrites Giskard trajectory length limit for planning.
        If the trajectory is longer than new_length, Giskard will prempt the goal.
        :param new_length: in seconds
        """
        self.monitors.add_max_trajectory_length(max_trajectory_length=new_length,
                                                **kwargs)

    def set_limit_cartesian_velocity_goal(self,
                                          tip_link: str,
                                          root_link: str,
                                          tip_group: Optional[str] = None,
                                          root_group: Optional[str] = None,
                                          max_linear_velocity: float = 0.1,
                                          max_angular_velocity: float = 0.5,
                                          weight: float = WEIGHT_ABOVE_CA,
                                          hard: bool = False,
                                          **kwargs: goal_parameter):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param root_group: if the root_link is not unique, use this to say to which group the link belongs
        :param tip_group: if the tip_link is not unique, use this to say to which group the link belongs
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        self.motion_goals.add_limit_cartesian_velocity(root_link=root_link,
                                                       root_group=root_group,
                                                       tip_link=tip_link,
                                                       tip_group=tip_group,
                                                       weight=weight,
                                                       max_linear_velocity=max_linear_velocity,
                                                       max_angular_velocity=max_angular_velocity,
                                                       hard=hard,
                                                       **kwargs)

    def set_grasp_bar_goal(self,
                           bar_center: PointStamped,
                           bar_axis: Vector3Stamped,
                           bar_length: float,
                           tip_link: str,
                           tip_grasp_axis: Vector3Stamped,
                           root_link: str,
                           tip_group: Optional[str] = None,
                           root_group: Optional[str] = None,
                           reference_linear_velocity: Optional[float] = None,
                           reference_angular_velocity: Optional[float] = None,
                           weight: float = WEIGHT_ABOVE_CA,
                           add_monitor: bool = True,
                           **kwargs: goal_parameter):
        """
        Like a CartesianPose but with more freedom.
        tip_link is allowed to be at any point along bar_axis, that is without bar_center +/- bar_length.
        It will align tip_grasp_axis with bar_axis, but allows rotation around it.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param tip_grasp_axis: axis of tip_link that will be aligned with bar_axis
        :param bar_center: center of the bar to be grasped
        :param bar_axis: alignment of the bar to be grasped
        :param bar_length: length of the bar to be grasped
        :param root_group: if root_link is not unique, search in this group for matches
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight:
        :param add_monitor: if True, adds a monitor as end_condition to check if the goal was reached.
        """
        end_condition = ''
        if add_monitor:
            monitor_name1 = self.monitors.add_distance_to_line(root_link=root_link,
                                                               tip_link=tip_link,
                                                               center_point=bar_center,
                                                               line_axis=bar_axis,
                                                               line_length=bar_length,
                                                               root_group=root_group,
                                                               tip_group=tip_group)
            monitor_name2 = self.monitors.add_vectors_aligned(root_link=root_link,
                                                              tip_link=tip_link,
                                                              goal_normal=bar_axis,
                                                              tip_normal=tip_grasp_axis,
                                                              root_group=root_group,
                                                              tip_group=tip_group)
            end_condition = f'{monitor_name1} and {monitor_name2}'
        self.motion_goals.add_grasp_bar(end_condition=end_condition,
                                        root_link=root_link,
                                        tip_link=tip_link,
                                        tip_grasp_axis=tip_grasp_axis,
                                        bar_center=bar_center,
                                        bar_axis=bar_axis,
                                        bar_length=bar_length,
                                        root_group=root_group,
                                        tip_group=tip_group,
                                        reference_linear_velocity=reference_linear_velocity,
                                        reference_angular_velocity=reference_angular_velocity,
                                        weight=weight,
                                        **kwargs)

    def set_open_container_goal(self,
                                tip_link: str,
                                environment_link: str,
                                special_door: Optional[bool] = False,
                                special_door_state: Optional[float] = 0.0,
                                tip_group: Optional[str] = None,
                                environment_group: Optional[str] = None,
                                goal_joint_state: Optional[float] = None,
                                start_condition: str = '',
                                hold_condition: str = '',
                                end_condition: str = '',
                                weight=WEIGHT_ABOVE_CA):
        """
        Open a container in an environment.
        Only works with the environment was added as urdf.
        Assumes that a handle has already been grasped.
        Can only handle containers with 1 dof, e.g. drawers or doors.
        :param tip_link: end effector that is grasping the handle
        :param environment_link: name of the handle that was grasped
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param environment_group: if environment_link is not unique, search in this group for matches
        :param goal_joint_state: goal state for the container. default is maximum joint state.
        :param weight:
        """
        self.motion_goals.add_open_container(tip_link=tip_link,
                                             environment_link=environment_link,
                                             tip_group=tip_group,
                                             special_door=special_door,
                                             special_door_state=special_door_state,
                                             environment_group=environment_group,
                                             goal_joint_state=goal_joint_state,
                                             start_condition=start_condition,
                                             hold_condition=hold_condition,
                                             end_condition=end_condition,
                                             weight=weight)

    def set_align_to_push_door_goal(self,
                                    root_link: str,
                                    tip_link: str,
                                    door_object: str,
                                    door_handle: str,
                                    tip_gripper_axis: Vector3Stamped,
                                    tip_group: Optional[str] = None,
                                    root_group: Optional[str] = None,
                                    intermediate_point_scale: Optional[float] = 1,
                                    weight: float = WEIGHT_BELOW_CA):
        """
        Aligns the tip_link with the door_object to push it open. Only works if the door object is part of the urdf.
        The door has to be open a little before aligning.
        : param root_link: root link of the kinematic chain
        : param tip_link: end effector
        : param door object: name of the object to be pushed
        : param door_handle: name of the object handle
        : param tip_gripper_axis: axis of the tip_link that will be aligned along the door rotation axis
        """
        self.motion_goals.add_align_to_push_door(root_link=root_link,
                                                 tip_link=tip_link,
                                                 door_object=door_object,
                                                 door_handle=door_handle,
                                                 intermediate_point_scale=intermediate_point_scale,
                                                 tip_group=tip_group,
                                                 tip_gripper_axis=tip_gripper_axis,
                                                 root_group=root_group,
                                                 weight=weight)

    def set_pre_push_door_goal(self,
                               root_link: str,
                               tip_link: str,
                               door_object: str,
                               door_handle: str,
                               reference_linear_velocity: Optional[float] = None,
                               reference_angular_velocity: Optional[float] = None,
                               weight: float = WEIGHT_ABOVE_CA):
        """
        Positions the gripper in contact with the door before pushing to open.
        : param root_link: root link of the kinematic chain
        : param tip_link: end effector
        : param door object: name of the object to be pushed
        : param door_height: height of the door
        : param door_length: length of the door
        : param root_V_object_rotation_axis: door rotation axis w.r.t root
        : param root_V_object_normal: door normal w.r.t root
        """
        self.motion_goals.add_pre_push_door(root_link=root_link,
                                            tip_link=tip_link,
                                            door_object=door_object,
                                            door_handle=door_handle,
                                            reference_linear_velocity=reference_linear_velocity,
                                            reference_angular_velocity=reference_angular_velocity,
                                            weight=weight)

    def set_close_container_goal(self,
                                 tip_link: str,
                                 environment_link: str,
                                 tip_group: Optional[str] = None,
                                 environment_group: Optional[str] = None,
                                 goal_joint_state: Optional[float] = None,
                                 weight: Optional[float] = None):
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        self.motion_goals.add_close_container(tip_link=tip_link,
                                              environment_link=environment_link,
                                              tip_group=tip_group,
                                              environment_group=environment_group,
                                              goal_joint_state=goal_joint_state,
                                              weight=weight)

    def set_pointing_goal(self,
                          goal_point: PointStamped,
                          tip_link: str,
                          pointing_axis: Vector3Stamped,
                          root_link: str,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          max_velocity: float = 0.3,
                          weight: float = WEIGHT_BELOW_CA,
                          add_monitor: bool = True,
                          **kwargs: goal_parameter):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param goal_point: where to point pointing_axis at.
        :param root_link: root link of the kinematic chain.
        :param tip_group: if tip_link is not unique, search this group for matches.
        :param root_group: if root_link is not unique, search this group for matches.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        :param weight:
        :param add_monitor: if True, adds a monitor as end_condition to check if the goal was reached.
        """
        if add_monitor:
            end_condition = self.monitors.add_pointing_at(goal_point=goal_point,
                                                          tip_link=tip_link,
                                                          pointing_axis=pointing_axis,
                                                          root_link=root_link,
                                                          tip_group=tip_group,
                                                          root_group=root_group)
        else:
            end_condition = ''
        self.motion_goals.add_pointing(end_condition=end_condition,
                                       tip_link=tip_link,
                                       tip_group=tip_group,
                                       goal_point=goal_point,
                                       root_link=root_link,
                                       root_group=root_group,
                                       pointing_axis=pointing_axis,
                                       max_velocity=max_velocity,
                                       weight=weight,
                                       **kwargs)

    def set_avoid_joint_limits_goal(self,
                                    percentage: int = 15,
                                    joint_list: Optional[List[str]] = None,
                                    weight: Optional[float] = None):
        """
        This goal will push joints away from their position limits. For example if percentage is 15 and the joint
        limits are 0-100, it will push it into the 15-85 range.
        """
        self.motion_goals.add_avoid_joint_limits(percentage=percentage,
                                                 weight=weight,
                                                 joint_list=joint_list)

    # %% collision avoidance
    def allow_collision(self,
                        group1: str = CollisionEntry.ALL,
                        group2: str = CollisionEntry.ALL):
        """
        Tell Giskard to allow collision between group1 and group2. Use CollisionEntry.ALL to allow collision with all
        groups.
        :param group1: name of the first group
        :param group2: name of the second group
        """
        self.motion_goals.allow_collision(group1=group1,
                                          group2=group2)

    def avoid_collision(self,
                        min_distance: Optional[float] = None,
                        group1: str = CollisionEntry.ALL,
                        group2: str = CollisionEntry.ALL):
        """
        Tell Giskard to avoid collision between group1 and group2. Use CollisionEntry.ALL to allow collision with all
        groups.
        :param min_distance: set this to overwrite the default distances
        :param group1: name of the first group
        :param group2: name of the second group
        """
        self.motion_goals.avoid_collision(min_distance=min_distance,
                                          group1=group1,
                                          group2=group2)

    def allow_all_collisions(self):
        """
        Allows all collisions for next goal.
        """
        self.motion_goals.allow_all_collisions()

    def avoid_all_collisions(self, min_distance: Optional[float] = None):
        """
        Avoids all collisions for next goal.
        If you don't want to override the distance, don't call this function. Avoid all is the default, if you don't
        add any collision entries.
        :param min_distance: set this to overwrite default distances
        """
        self.motion_goals.avoid_all_collisions(min_distance=min_distance)

    def allow_self_collision(self, robot_name: Optional[str] = None):
        """
        Allows the collision of the robot with itself for the next goal.
        :param robot_name: if there are multiple robots, specify which one.
        """
        self.motion_goals.allow_self_collision(robot_name=robot_name)

    def add_box(self,
                name: str,
                size: Tuple[float, float, float],
                pose: PoseStamped,
                parent_link: str = '',
                parent_link_group: str = '') -> WorldResult:
        """
        Adds a new box to the world tree and attaches it to parent_link.
        If parent_link_group and parent_link are empty, the box will be attached to the world root link, e.g., map.
        :param name: How the new group will be called
        :param size: X, Y and Z dimensions of the box, respectively
        :param pose: Where the root link of the new object will be positioned
        :param parent_link: Name of the link, the object will get attached to
        :param parent_link_group: Name of the group in which Giskard will search for parent_link
        :return: Response message of the service call
        """
        return self.world.add_box(name=name,
                                  size=size,
                                  pose=pose,
                                  parent_link=parent_link,
                                  parent_link_group=parent_link_group)

    def add_sphere(self,
                   name: str,
                   radius: float,
                   pose: PoseStamped,
                   parent_link: str = '',
                   parent_link_group: str = '') -> WorldResult:
        """
        See add_box.
        """
        return self.world.add_sphere(name=name,
                                     radius=radius,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group)

    def add_mesh(self,
                 name: str,
                 mesh: str,
                 pose: PoseStamped,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 scale: Tuple[float, float, float] = (1, 1, 1)) -> WorldResult:
        """
        See add_box.
        :param mesh: path to the mesh location, can be ros package path, e.g.,
                        package://giskardpy/test/urdfs/meshes/bowl_21.obj
        """
        return self.world.add_mesh(name=name,
                                   mesh=mesh,
                                   scale=scale,
                                   pose=pose,
                                   parent_link=parent_link,
                                   parent_link_group=parent_link_group)

    def add_cylinder(self,
                     name: str,
                     height: float,
                     radius: float,
                     pose: PoseStamped,
                     parent_link: str = '',
                     parent_link_group: str = '') -> WorldResult:
        """
        See add_box.
        """
        return self.world.add_cylinder(name=name,
                                       height=height,
                                       radius=radius,
                                       pose=pose,
                                       parent_link=parent_link,
                                       parent_link_group=parent_link_group)

    def remove_group(self, name: str) -> WorldResult:
        """
        Removes a group and all links and joints it contains from the world.
        Be careful, you can remove parts of the robot like that.
        """
        return self.world.remove_group(name=name)

    def update_parent_link_of_group(self,
                                    name: str,
                                    parent_link: str,
                                    parent_link_group: Optional[str] = '') -> WorldResult:
        """
        Removes the joint connecting the root link of a group and attaches it to a parent_link.
        The object will not move relative to the world's root link in this process.
        :param name: name of the group
        :param parent_link: name of the new parent link
        :param parent_link_group: if parent_link is not unique, search in this group for matches.
        :return: result message
        """
        return self.world.update_parent_link_of_group(name=name,
                                                      parent_link=parent_link,
                                                      parent_link_group=parent_link_group)

    def detach_group(self, object_name: str):
        """
        A wrapper for update_parent_link_of_group which set parent_link to the root link of the world.
        """
        return self.world.detach_group(oname=object_name)

    def add_urdf(self,
                 name: str,
                 urdf: str,
                 pose: PoseStamped,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 js_topic: Optional[str] = '') -> WorldResult:
        """
        Adds a urdf to the world.
        :param name: name the group containing the urdf will have.
        :param urdf: urdf as string, no path!
        :param pose: pose of the root link of the new object
        :param parent_link: to which link the urdf will be attached
        :param parent_link_group: if parent_link is not unique, search here for matches.
        :param js_topic: Giskard will listen on that topic for joint states and update the urdf accordingly
        :return: response message
        """
        return self.world.add_urdf(name=name,
                                   urdf=urdf,
                                   pose=pose,
                                   js_topic=js_topic,
                                   parent_link=parent_link,
                                   parent_link_group=parent_link_group)

    def dye_group(self, group_name: str, rgba: Tuple[float, float, float, float]) -> DyeGroupResponse:
        """
        Change the color of the ghost for this particular group.
        """
        return self.world.dye_group(group_name=group_name, rgba=rgba)

    def get_group_names(self) -> List[str]:
        """
        Returns the names of every group in the world.
        """
        return self.world.get_group_names()

    def get_group_info(self, group_name: str) -> GetGroupInfoResponse:
        """
        Returns the joint state, joint state topic and pose of a group.
        """
        return self.world.get_group_info(group_name=group_name)

    def get_controlled_joints(self, group_name: str) -> List[str]:
        """
        Returns all joints of a group that are flagged as controlled.
        """
        return self.world.get_controlled_joints(group_name=group_name)

    def update_group_pose(self, group_name: str, new_pose: PoseStamped) -> WorldResult:
        """
        Overwrites the pose specified in the joint that connects the two groups.
        :param group_name: Name of the group that will move
        :param new_pose: New pose of the group
        :return: Giskard's reply
        """
        return self.world.update_group_pose(group_name=group_name, new_pose=new_pose)

    def register_group(self, new_group_name: str, root_link_name: str,
                       root_link_group_name: str) -> WorldResult:
        """
        Register a new group for reference in collision checking. All child links of root_link_name will belong to it.
        :param new_group_name: Name of the new group.
        :param root_link_name: root link of the new group
        :param root_link_group_name: Name of the group root_link_name belongs to
        :return: WorldResult
        """
        return self.world.register_group(new_group_name=new_group_name,
                                         root_link_name=root_link_name,
                                         root_link_group_name=root_link_group_name)

    def clear_world(self) -> WorldResult:
        """
        Resets the world to what it was when Giskard was launched.
        """
        return self.world.clear()

    def move_gripper(self,
                     gripper_state: str):

        self.motion_goals.add_motion_goal(motion_goal_class='MoveGripper',
                                          gripper_state=gripper_state)

    def reaching(self,
                 grasp: str,
                 align: str,
                 object_name: str,
                 object_shape: str,
                 goal_pose: Optional[PoseStamped] = None,
                 object_size: Optional[Vector3] = None,
                 root_link: str = 'map',
                 tip_link: str = 'hand_palm_link',
                 velocity: float = 0.2):
        """
        :param grasp: direction to grasp from, directions are: front, top, right, left, below
        :param align: the frame that the wrist frame aligns with, will be ignored if left empty
        :param object_name: name of object that should be reached
        :param object_shape: shape of the object (current options are cylinder, sphere and rectangle)
        :param goal_pose: position of the goal that should be reached
        :param object_size: size of the object as a Vector3 (in meters)
        :param root_link: the root link, usually map
        :param tip_link: the tip link, normally hand_palm_link
        :param velocity: velocity of executed movement
        """

        self.motion_goals.add_motion_goal(motion_goal_class=Reaching.__name__,
                                          grasp=grasp,
                                          align=align,
                                          object_name=object_name,
                                          object_shape=object_shape,
                                          goal_pose=goal_pose,
                                          object_size=object_size,
                                          root_link=root_link,
                                          tip_link=tip_link,
                                          velocity=velocity)

    def placing(self,
                context,
                goal_pose: PoseStamped,
                tip_link: str = 'hand_palm_link',
                velocity: float = 0.02):

        self.motion_goals.add_motion_goal(motion_goal_class=Placing.__name__,
                                          context=context,
                                          goal_pose=goal_pose,
                                          tip_link=tip_link,
                                          velocity=velocity)

    def vertical_motion(self,
                        context: str,
                        distance: float = 0.02,
                        root_link: str = 'base_link',
                        tip_link: str = 'hand_palm_link'):

        self.motion_goals.add_motion_goal(motion_goal_class='VerticalMotion',
                                          context=context,
                                          distance=distance,
                                          root_link=root_link,
                                          tip_link=tip_link)

    def retract(self,
                object_name: str,
                distance: float = 0.1,
                reference_frame: str = 'base_link',
                root_link: str = 'map',
                tip_link: str = 'base_link',
                velocity: float = 0.2):

        self.motion_goals.add_motion_goal(motion_goal_class=Retracting.__name__,
                                          object_name=object_name,
                                          distance=distance,
                                          reference_frame=reference_frame,
                                          root_link=root_link,
                                          tip_link=tip_link,
                                          velocity=velocity)

    def align_height(self,
                     context,
                     object_name: str,
                     goal_pose: PoseStamped,
                     object_height: float,
                     root_link: str = 'map',
                     tip_link: str = 'hand_gripper_tool_frame'):

        self.motion_goals.add_motion_goal(motion_goal_class='AlignHeight',
                                          context=context,
                                          object_name=object_name,
                                          goal_pose=goal_pose,
                                          object_height=object_height,
                                          root_link=root_link,
                                          tip_link=tip_link)

    def sequence_goal(self,
                      motion_sequence):

        self.motion_goals.add_motion_goal(motion_goal_class='SequenceGoal',
                                          motion_sequence=motion_sequence)

    def test_goal(self,
                  goal_name: str,
                  **kwargs):

        self.motion_goals.add_motion_goal(motion_goal_class=goal_name,
                                          **kwargs)

    def take_pose(self,
                  pose_keyword: str):

        self.motion_goals.add_motion_goal(motion_goal_class=TakePose.__name__,
                                          pose_keyword=pose_keyword)

    def tilting(self,
                tilt_direction: Optional[str] = None,
                tilt_angle: Optional[float] = None,
                tip_link: str = 'wrist_roll_joint',
                ):

        self.motion_goals.add_motion_goal(motion_goal_class=Tilting.__name__,
                                          direction=tilt_direction,
                                          angle=tilt_angle,
                                          tip_link=tip_link)

    def joint_rotation_continuous(self,
                                  joint_name: str,
                                  joint_center: float,
                                  joint_range: float,
                                  trajectory_length: float = 20,
                                  target_speed: float = 1,
                                  period_length: float = 1.0):
        self.motion_goals.add_motion_goal(motion_goal_class='JointRotationGoalContinuous',
                                          joint_name=joint_name,
                                          joint_center=joint_center,
                                          joint_range=joint_range,
                                          trajectory_length=trajectory_length,
                                          target_speed=target_speed,
                                          period_length=period_length)

    def mixing(self,
               mixing_time=20,
               weight: float = WEIGHT_ABOVE_CA):

        self.motion_goals.add_motion_goal(motion_goal_class='Mixing',
                                          mixing_time=mixing_time,
                                          weight=weight)

    def open_environment(self,
                         tip_link: str,
                         environment_link: str,
                         tip_group: Optional[str] = None,
                         environment_group: Optional[str] = None,
                         goal_joint_state: Optional[float] = None,
                         weight: float = WEIGHT_ABOVE_CA):

        self.motion_goals.add_motion_goal(motion_goal_class='Open',
                                          tip_link=tip_link,
                                          environment_link=environment_link,
                                          tip_group=tip_group,
                                          environment_group=environment_group,
                                          goal_joint_state=goal_joint_state,
                                          weight=weight)

    def push_button(self,
                    goal_pose,
                    tip_link,
                    velocity):
        self.motion_goals.add_motion_goal(motion_goal_class='PushButton',
                                          goal_pose=goal_pose,
                                          tip_link=tip_link,
                                          velocity=velocity)

    def move_base(self, target_pose: PoseStamped):
        """
        moving the hsr through the move_base interface from Toyota
        :param target_pose: the pose where robot moves to
        """
        cli = actionlib.SimpleActionClient('/move_base/move', MoveBaseAction)

        cli.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose = target_pose

        cli.send_goal(goal)

        cli.wait_for_result()

        action_state = cli.get_state()
        if action_state == GoalStatus.SUCCEEDED:
            rospy.loginfo("Navigation Succeeded")

    def change_gripper_state(self, gripper_state: str):
        """
        Rework proposal for the gripper,
        Now uses Enums via suturo_types.py, which in case of this function acts
        as a list of possible gripping commands. This also makes it possible
        to add gripping forces for specific object types.
        :param gripper_state: the state that the gripper shall assume
        """
        if self.is_standalone():
            if gripper_state == GripperTypes.OPEN.value:
                self.set_joint_goal({
                    'hand_motor_joint': 1.2
                })
                self.execute()

            elif gripper_state == GripperTypes.CLOSE.value:
                self.set_joint_goal({
                    'hand_motor_joint': 0.0
                })
                self.execute()

            elif gripper_state == GripperTypes.NEUTRAL.value:
                self.set_joint_goal({
                    'hand_motor_joint': 0.6
                })
                self.execute()
            else:
                rospy.logwarn("gripper_state {} not found".format(gripper_state))
        else:
            if gripper_state == GripperTypes.OPEN.value:
                self._move_gripper_force(0.8)

            elif gripper_state == GripperTypes.CLOSE.value:
                self._move_gripper_force(-0.8)

            elif gripper_state == GripperTypes.NEUTRAL.value:
                self._set_gripper_joint_position(0.5)
            else:
                rospy.logwarn("gripper_state {} not found".format(gripper_state))

    def _move_gripper_force(self, force: float = 0.8):
        """
        Closes the gripper with the given force.
        :param force: force to grasp which should be between 0.2 and 0.8 (N)
        :return: applied effort
        """
        _gripper_apply_force_client = actionlib.SimpleActionClient('/hsrb/gripper_controller/grasp',
                                                                   GripperApplyEffortAction)

        try:
            if not _gripper_apply_force_client.wait_for_server(rospy.Duration(
                    10)):
                raise Exception('/hsrb/gripper_controller/grasp does not exist')
        except Exception as e:
            rospy.logerr(e)
            return

        rospy.loginfo("Closing gripper with force: {}".format(force))
        f = force  # max(min(0.8, force), 0.2)
        goal = GripperApplyEffortGoal()
        goal.effort = f
        _gripper_apply_force_client.send_goal(goal)

    def _set_gripper_joint_position(self, position):
        """
        Sets the gripper joint to the given position
        :param position: goal position of the joint -0.105 to 1.239 rad
        :return: error_code of FollowJointTrajectoryResult
        """
        _gripper_controller = actionlib.SimpleActionClient('/hsrb/gripper_controller/follow_joint_trajectory',
                                                           FollowJointTrajectoryAction)

        # Wait for connection
        try:
            if not _gripper_controller.wait_for_server(rospy.Duration(10)):
                raise Exception('/hsrb/gripper_controller/follow_joint_trajectory does not exist')
        except Exception as e:
            rospy.logerr(e)

        pos = max(min(1.239, position), -0.105)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = [u'hand_motor_joint']
        p = JointTrajectoryPoint()
        p.positions = [pos]
        p.velocities = [0]
        p.effort = [0.1]
        p.time_from_start = rospy.Time(1)
        goal.trajectory.points = [p]
        _gripper_controller.send_goal(goal)

    def is_standalone(self) -> bool:
        return self.world.get_control_mode() == ControlModes.standalone

    def real_time_pointer(self, tip_link, topic_name, root_link, pointing_axis, endless_mode):
        """
        Wrapper for RealTimePointing and EndlessMode,
        which is used for person live-tracking.
        """
        # if endless_mode:
        #     self.motion_goals.add_motion_goal(motion_goal_class='EndlessMode')f

        self.motion_goals.add_motion_goal(motion_goal_class=RealTimePointingPose.__name__,
                                          tip_link=tip_link,
                                          topic_name=topic_name,
                                          root_link=root_link,
                                          pointing_axis=pointing_axis)

    def continuous_pointing_head(self):
        """
        Uses real_time_pointer for continuous tracking of a human_pose.
        """
        tip_V_pointing_axis: Vector3Stamped = Vector3Stamped()
        tip_V_pointing_axis.header.frame_id = 'head_center_camera_frame'
        tip_V_pointing_axis.vector.z = 1

        self.real_time_pointer(root_link='map',
                               tip_link='head_center_camera_frame',
                               topic_name='human_pose',
                               endless_mode=True,
                               pointing_axis=tip_V_pointing_axis)

    def check_controllers_active(self,
                                 stopped_controllers: Optional[List] = None,
                                 running_controllers: Optional[List] = None):
        """
        Checks if the arm_trajectory_controller and head_trajectory_controller are stopped
        and the realtime_body_controller_real is running

        :param stopped_controllers: controllers that have to be in state stopped or initialized
        :param running_controllers: controllers that have to be in state running
        """
        if stopped_controllers is None:
            stopped_controllers = []
        if running_controllers is None:
            running_controllers = []

        resp: ListControllersResponse = self.list_controller_srv()
        controller_dict = {controller.name: controller for controller in resp.controller}

        if (all(controller_dict[con].state == 'stopped' or controller_dict[con].state == 'initialized' for con in
                stopped_controllers) and all(controller_dict[con].state == 'running' for con in running_controllers)):
            return True
        return False

    def set_closed_loop_controllers(self,
                                    stop: Optional[List] = None,
                                    start: Optional[List] = None,
                                    switch_controller_srv: Optional[
                                        str] = '/hsrb/controller_manager/switch_controller') -> SwitchControllerResponse:
        """
        Start and Stop controllers for via the designated switch_controller service
        """

        if stop is None:
            stop = ['arm_trajectory_controller', 'head_trajectory_controller']
        if start is None:
            start = ['realtime_body_controller_real']

        start_controllers = start
        stop_controllers = stop
        strictness: int = 1
        start_asap: bool = False
        timeout: float = 0.0

        rospy.wait_for_service(switch_controller_srv)
        srv_switch_con = rospy.ServiceProxy(name=switch_controller_srv,
                                            service_class=SwitchController)

        resp: SwitchControllerResponse = srv_switch_con(start_controllers,
                                                        stop_controllers,
                                                        strictness,
                                                        start_asap,
                                                        timeout)
        return resp

    def set_open_door_goal(self,
                           tip_link: str,
                           door_handle_link: str,
                           name: str = None):
        """
        Adds OpenDoorGoal to motion goal execution plan

        :param tip_link: Link that is grasping the door handle
        :param door_handle_link: Link of the door handle of the door that is to be opened
        :param name: Name of the Goal for distinction between similar goals
        """
        self.motion_goals.add_motion_goal(motion_goal_class=OpenDoorGoal.__name__,
                                          tip_link=tip_link,
                                          door_handle_link=door_handle_link,
                                          name=name)

    def set_hsrb_open_door_goal(self,
                                door_handle_link: str,
                                tip_link: str = 'hand_gripper_tool_frame',
                                name: str = 'HSRB_open_door'):
        """
        HSRB specific open door goal wrapper

        :param door_handle_link: Link of the door handle
        :param tip_link: Link that's grasping the door handle
        :param name: name of the goal for distinction between same goals
        """

        self.set_open_door_goal(tip_link=tip_link,
                                door_handle_link=door_handle_link,
                                name=name)

    def set_hsrb_door_handle_grasp(self,
                                   handle_name: str,
                                   handle_bar_length: float = 0,
                                   tip_link: str = 'hand_gripper_tool_frame',
                                   root_link: str = 'map',
                                   bar_axis_v: Vector3 = Vector3(0, 1, 0),
                                   tip_grasp_axis_v: Vector3 = Vector3(1, 0, 0)):
        """
        HSRB specific set_grasp_bar_goal, that only needs handle_name of the door_handle

        :param handle_name: URDF link that represents the door handle
        :param handle_bar_length: length of the door handle
        :param tip_link: robot link, that grasps the handle
        :param root_link: root link of the kinematic chain
        :param bar_axis_v: Vector for changing the orientation of the door handle
        :param tip_grasp_axis_v: Vector for the orientation of the tip grasp link
        """
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_name
        bar_axis.vector = bar_axis_v

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_name
        bar_center.point.y = 0.045

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = tip_link
        tip_grasp_axis.vector = tip_grasp_axis_v

        self.set_grasp_bar_goal(root_link=root_link,
                                tip_link=tip_link,
                                tip_grasp_axis=tip_grasp_axis,
                                bar_center=bar_center,
                                bar_axis=bar_axis,
                                bar_length=handle_bar_length)

    def set_hsrb_dishwasher_door_around(self,
                                        handle_name: str,
                                        root_link: str = 'map',
                                        tip_link: str = 'hand_gripper_tool_frame',
                                        door_hinge_frame_id: Optional[str] = None,
                                        points_negative_to_positive: Optional[bool] = True):
        """
        HSRB specific avoid dishwasher door goal

        :param handle_name: name of the door handle
        :param tip_link: robot link, that grasps the handle
        :param root_link: root link of the kinematic chain
        :param door_hinge_frame_id: frame id of the door hinge
        :param points_negative_to_positive: boolean to determine if points go from negative angle offset
                                            to positive angle offset
        """

        self.motion_goals.add_motion_goal(motion_goal_class=MoveAroundDoor.__name__,
                                          handle_name=handle_name,
                                          root_link=root_link,
                                          tip_link=tip_link,
                                          door_hinge_frame_id=door_hinge_frame_id,
                                          points_negative_to_positive=points_negative_to_positive)

    def set_hsrb_align_to_push_door_goal(self,
                                         handle_name: str,
                                         hinge_frame_id: str,
                                         tip_link: str = 'hand_gripper_tool_frame',
                                         root_link: str = 'map'):
        """
        HSRB specific push door open goal of dishwasher

        :param handle_name: name of the door handle
        :param hinge_frame_id: Frame id of the door hinge
        :param tip_link: robot link, that grasps the handle
        :param root_link: root link of the kinematic chain
        """

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = tip_link
        tip_grasp_axis.vector.y = 1

        self.set_align_to_push_door_goal(root_link=root_link,
                                         tip_link=tip_link,
                                         door_handle=handle_name,
                                         door_object=hinge_frame_id,
                                         tip_gripper_axis=tip_grasp_axis,
                                         intermediate_point_scale=0.95)

    def set_hsrb_pre_push_door_goal(self,
                                    handle_name: str,
                                    hinge_frame_id: str,
                                    root_link: str = 'map',
                                    tip_link: str = 'hand_gripper_tool_frame'):
        """
        HSRB specific pre push door open goal of dishwasher

        :param handle_name: name of the door handle
        :param hinge_frame_id: Frame id of the door hinge
        :param tip_link: robot link, that grasps the handle
        :param root_link: root link of the kinematic chain
        """

        self.set_pre_push_door_goal(root_link=root_link,
                                    tip_link=tip_link,
                                    door_handle=handle_name,
                                    door_object=hinge_frame_id)

    def set_hsrb_dishwasher_door_handle_grasp(self,
                                              handle_frame_id: str,
                                              grasp_bar_offset: float = 0.0,
                                              root_link: str = 'map',
                                              tip_link: str = 'hand_gripper_tool_frame'):

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = tip_link
        tip_grasp_axis.vector.x = 1

        grasp_axis_offset = Vector3Stamped()
        grasp_axis_offset.header.frame_id = handle_frame_id
        grasp_axis_offset.vector.x = -grasp_bar_offset

        self.set_grasp_bar_offset_goal(root_link=root_link,
                                       tip_link=tip_link,
                                       tip_grasp_axis=tip_grasp_axis,
                                       bar_center=bar_center,
                                       bar_axis=bar_axis,
                                       bar_length=.4,
                                       grasp_axis_offset=grasp_axis_offset)

        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = tip_link
        x_gripper.vector.z = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1

        self.set_align_planes_goal(tip_link=tip_link,
                                   tip_normal=x_gripper,
                                   goal_normal=x_goal,
                                   root_link=root_link)

    def set_grasp_bar_offset_goal(self,
                                  bar_center: PointStamped,
                                  bar_axis: Vector3Stamped,
                                  bar_length: float,
                                  tip_link: str,
                                  tip_grasp_axis: Vector3Stamped,
                                  root_link: str,
                                  grasp_axis_offset: Vector3Stamped,
                                  tip_group: Optional[str] = None,
                                  root_group: Optional[str] = None,
                                  reference_linear_velocity: Optional[float] = None,
                                  reference_angular_velocity: Optional[float] = None,
                                  weight: float = WEIGHT_ABOVE_CA,
                                  add_monitor: bool = True,
                                  **kwargs: goal_parameter):
        """
        Like a CartesianPose but with more freedom.
        tip_link is allowed to be at any point along bar_axis, that is without bar_center +/- bar_length.
        It will align tip_grasp_axis with bar_axis, but allows rotation around it.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param tip_grasp_axis: axis of tip_link that will be aligned with bar_axis
        :param bar_center: center of the bar to be grasped
        :param bar_axis: alignment of the bar to be grasped
        :param bar_length: length of the bar to be grasped
        :param grasp_axis_offset: offset of the tip_link to the bar_center
        :param root_group: if root_link is not unique, search in this group for matches
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight:
        :param add_monitor: if True, adds a monitor as end_condition to check if the goal was reached.
        """
        end_condition = ''
        if add_monitor:
            monitor_name1 = self.monitors.add_distance_to_line(root_link=root_link,
                                                               tip_link=tip_link,
                                                               center_point=bar_center,
                                                               line_axis=bar_axis,
                                                               line_length=bar_length,
                                                               root_group=root_group,
                                                               tip_group=tip_group)
            monitor_name2 = self.monitors.add_vectors_aligned(root_link=root_link,
                                                              tip_link=tip_link,
                                                              goal_normal=bar_axis,
                                                              tip_normal=tip_grasp_axis,
                                                              root_group=root_group,
                                                              tip_group=tip_group)
            end_condition = f'{monitor_name1} and {monitor_name2}'
        self.motion_goals.add_grasp_bar_offset(end_condition=end_condition,
                                               root_link=root_link,
                                               tip_link=tip_link,
                                               tip_grasp_axis=tip_grasp_axis,
                                               bar_center=bar_center,
                                               bar_axis=bar_axis,
                                               bar_length=bar_length,
                                               grasp_axis_offset=grasp_axis_offset,
                                               root_group=root_group,
                                               tip_group=tip_group,
                                               reference_linear_velocity=reference_linear_velocity,
                                               reference_angular_velocity=reference_angular_velocity,
                                               weight=weight,
                                               **kwargs)
