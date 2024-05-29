from __future__ import division

from typing import Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.goals.cartesian_goals import CartesianPosition, CartesianOrientation
from giskardpy.goals.goal import Goal
from giskardpy.goals.joint_goals import JointPositionList
from giskardpy.god_map import god_map
from giskardpy.monitors.joint_monitors import JointGoalReached
from giskardpy.monitors.payload_monitors import Sleep
from giskardpy.tasks.task import WEIGHT_ABOVE_CA
from giskardpy.tasks.task import WEIGHT_BELOW_CA


class Open(Goal):
    def __init__(self,
                 tip_link: str,
                 environment_link: str,
                 special_door: Optional[bool] = False,
                 special_door_state: Optional[float] = 0.0,
                 tip_group: Optional[str] = None,
                 environment_group: Optional[str] = None,
                 goal_joint_state: Optional[float] = None,
                 max_velocity: float = 100,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
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
        self.weight = weight
        self.tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        self.handle_link = god_map.world.search_for_link_name(environment_link, environment_group)
        self.joint_name = god_map.world.get_movable_parent_joint(self.handle_link)
        self.handle_T_tip = god_map.world.compute_fk_pose(self.handle_link, self.tip_link)
        if name is None:
            name = f'{self.__class__.__name__}'
        super().__init__(name)

        _, max_position = god_map.world.get_joint_position_limits(self.joint_name)
        if goal_joint_state is None:
            goal_joint_state = max_position
        else:
            goal_joint_state = min(max_position, goal_joint_state)
            # goal_joint_state = goal_joint_state

        if not cas.is_true(start_condition):
            handle_T_tip = god_map.world.compose_fk_expression(self.handle_link, self.tip_link)
            handle_T_tip = god_map.monitor_manager.register_expression_updater(handle_T_tip,
                                                                               start_condition)
        else:
            handle_T_tip = cas.TransMatrix(god_map.world.compute_fk_pose(self.handle_link, self.tip_link))

        if special_door:
            monitor_goal_state = {self.joint_name: goal_joint_state - 0.1}

            door_joint_state_monitor = JointGoalReached(goal_state=monitor_goal_state,
                                                        threshold=0.05,
                                                        name=f'{name}_door_joint_monitor')
            self.add_monitor(door_joint_state_monitor)

            end_condition = cas.logic_or(end_condition, door_joint_state_monitor.get_state_expression())

            god_map.debug_expression_manager.add_debug_expression('joint_state',
                                                                  door_joint_state_monitor.get_state_expression())

        # %% position
        r_P_c = god_map.world.compose_fk_expression(self.handle_link, self.tip_link).to_position()
        task = self.create_and_add_task('position')
        task.add_point_goal_constraints(frame_P_goal=handle_T_tip.to_position(),
                                        frame_P_current=r_P_c,
                                        reference_velocity=CartesianPosition.default_reference_velocity,
                                        weight=self.weight)

        # %% orientation
        r_R_c = god_map.world.compose_fk_expression(self.handle_link, self.tip_link).to_rotation()
        c_R_r_eval = god_map.world.compose_fk_evaluated_expression(self.tip_link, self.handle_link).to_rotation()

        task = self.create_and_add_task('orientation')
        task.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=handle_T_tip.to_rotation(),
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=CartesianOrientation.default_reference_velocity,
                                           weight=self.weight)

        self.connect_monitors_to_all_tasks(start_condition=start_condition,
                                           hold_condition=hold_condition,
                                           end_condition=end_condition)

        goal_state = {self.joint_name.short_name: goal_joint_state}

        self.add_constraints_of_goal(JointPositionList(goal_state=goal_state,
                                                       max_velocity=max_velocity,
                                                       weight=WEIGHT_BELOW_CA,
                                                       start_condition=start_condition,
                                                       hold_condition=hold_condition,
                                                       end_condition=end_condition))

        if goal_joint_state > 1 and special_door:
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.0
            arm_flex_joint = 0.0
            arm_roll_joint = -1.5
            wrist_flex_joint = -1.5
            wrist_roll_joint = 0.0

            joint_states = {
                'head_pan_joint': head_pan_joint,
                'head_tilt_joint': head_tilt_joint,
                'arm_lift_joint': arm_lift_joint,
                'arm_flex_joint': arm_flex_joint,
                'arm_roll_joint': arm_roll_joint,
                'wrist_flex_joint': wrist_flex_joint,
                'wrist_roll_joint': wrist_roll_joint}

            sleep_mon = Sleep(seconds=3,
                              start_condition=end_condition)
            self.add_monitor(sleep_mon)

            self.add_constraints_of_goal(JointPositionList(goal_state=joint_states,
                                                           start_condition=end_condition,
                                                           hold_condition=hold_condition,
                                                           end_condition=sleep_mon.get_state_expression()))


class Close(Goal):
    def __init__(self,
                 tip_link: str,
                 environment_link: str,
                 tip_group: Optional[str] = None,
                 environment_group: Optional[str] = None,
                 goal_joint_state: Optional[float] = None,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        self.tip_link = tip_link
        self.environment_link = environment_link
        if name is None:
            name = f'{self.__class__.__name__}'
        super().__init__(name)
        handle_link = god_map.world.search_for_link_name(environment_link, environment_group)
        joint_name = god_map.world.get_movable_parent_joint(handle_link)
        min_position, _ = god_map.world.get_joint_position_limits(joint_name)
        if goal_joint_state is None:
            goal_joint_state = min_position
        else:
            goal_joint_state = max(min_position, goal_joint_state)
        self.add_constraints_of_goal(Open(tip_link=tip_link,
                                          tip_group=tip_group,
                                          environment_link=environment_link,
                                          environment_group=environment_group,
                                          goal_joint_state=goal_joint_state,
                                          weight=weight,
                                          start_condition=start_condition,
                                          hold_condition=hold_condition,
                                          end_condition=end_condition))
