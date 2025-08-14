from typing import Optional, List, Tuple

import numpy as np
from giskardpy import casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName, ColorRGBA
from giskardpy.data_types.suturo_types import MoveAroundHingeAlign
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.monitors.monitors import Monitor
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA, Task


class MoveAroundHinge(Goal):
    old_position_monitor: Monitor = None
    tip_gripper_axis: cas.Vector3 = None
    root_V_tip_grasp_axis: cas.Vector3 = None
    root_V_object_rotation_axis: cas.Vector3 = None

    def __init__(self,
                 handle_name: PrefixName,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 tip_gripper_axis: cas.Vector3 = None,
                 reference_linear_velocity: float = 0.1,
                 reference_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_BELOW_CA,
                 goal_angle: float = None,
                 multipliers: Optional[List[Tuple[float, float, str]]] = None,
                 offset: Optional[cas.Vector3] = None,
                 align_gripper: MoveAroundHingeAlign = MoveAroundHingeAlign.LAST,
                 name: str = None):
        """
        Adds Points to move around the hinge to a given handle

        :param handle_name: full frame id of the handle
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param reference_linear_velocity: m/s
        :param weight:
        :param name: Name of the goal
        """
        if name is None:
            name = 'MoveAroundHingeGoal'
        super().__init__(name=name)
        if multipliers is None:
            multipliers = [(11 / 10, -0.7, 'down_short'),
                           (7 / 5, -0.3, 'down_long'),
                           (7 / 5, 0.4, 'up_long')]
        if offset is None:
            offset = cas.Vector3().from_xyz(0, 0, 0, root_link)

        self.weight = weight
        self.reference_linear_velocity = reference_linear_velocity
        self.reference_angular_velocity = reference_angular_velocity

        self.handle_frame_id = self.tip_link = god_map.world.search_for_link_name(handle_name)

        hinge_joint = god_map.world.get_movable_parent_joint(self.handle_frame_id)
        door_hinge_frame_id = god_map.world.get_parent_link_of_link(self.handle_frame_id)

        self.tip_link = tip_link
        self.root_link = root_link

        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        root_P_tip = root_T_tip.to_position()
        object_joint_angle = god_map.world.state[hinge_joint].position

        object_V_object_rotation_axis = cas.Vector3(god_map.world.get_joint(hinge_joint).axis)
        root_T_door_expr = god_map.world.compose_fk_expression(self.root_link, door_hinge_frame_id)
        root_V_offset = god_map.world.transform(self.root_link, offset)
        root_T_offset = cas.TransMatrix().from_xyz_rpy(root_V_offset.x,
                                                       root_V_offset.y,
                                                       root_V_offset.z,
                                                       reference_frame=offset.reference_frame)
        root_T_door_expr = root_T_offset.dot(root_T_door_expr)

        if tip_gripper_axis is not None:
            tip_gripper_axis.scale(1)
            self.tip_gripper_axis = tip_gripper_axis

            tip_V_tip_grasp_axis = cas.Vector3(self.tip_gripper_axis)
            self.root_V_tip_grasp_axis = cas.dot(root_T_tip, tip_V_tip_grasp_axis)
            self.root_V_object_rotation_axis = cas.dot(root_T_door_expr, object_V_object_rotation_axis)

        door_P_handle = god_map.world.compute_fk(door_hinge_frame_id, self.handle_frame_id).to_position()
        temp_point = door_P_handle.to_np()
        # axis pointing in the direction of handle frame from door joint frame
        direction_axis = np.argmax(abs(temp_point[0:3]))

        root_P_top_chain = []

        for i, (axis_multi, angle_multi, goal_name) in enumerate(multipliers):
            door_P_intermediate_point = np.zeros(3)
            door_P_intermediate_point[direction_axis] = temp_point[direction_axis] * axis_multi
            door_P_intermediate_point = cas.Point3([door_P_intermediate_point[0],
                                                    door_P_intermediate_point[1],
                                                    door_P_intermediate_point[2]])

            # # point w.r.t door
            if goal_angle is None:
                desired_angle = object_joint_angle * angle_multi  # just chose 1/2 of the goal angle
            else:
                desired_angle = goal_angle * angle_multi

            # find point w.r.t rotated door in local frame
            door_R_door_rotated = cas.RotationMatrix.from_axis_angle(axis=object_V_object_rotation_axis,
                                                                     angle=desired_angle)
            door_T_door_rotated = cas.TransMatrix(door_R_door_rotated)
            # as the root_T_door is already pointing to a completely rotated door, we invert desired angle to get to the
            # intermediate point
            door_rotated_P_top = cas.dot(door_T_door_rotated.inverse(), door_P_intermediate_point)
            root_P_top = cas.dot(cas.TransMatrix(root_T_door_expr), door_rotated_P_top)

            root_P_top_chain.append((root_P_top, goal_name))

        for i, (root_P_top, goal_name) in enumerate(root_P_top_chain):
            god_map.debug_expression_manager.add_debug_expression(f'goal_point_{goal_name}', root_P_top,
                                                                  color=ColorRGBA(0, 0.5, 0.5, 1))

            task = Task(name=goal_name)
            self.add_task(task)

            if self.old_position_monitor is None:
                position_monitor = Monitor(name=f'{goal_name}_pos_monitor')
            else:
                position_monitor = Monitor(name=f'{goal_name}_pos_monitor')
                position_monitor.start_condition = self.old_position_monitor

            distance_to_point = cas.euclidean_distance(root_P_tip, root_P_top)
            point_reached = cas.less(distance_to_point, 0.01)
            position_monitor.observation_expression = point_reached
            self.add_monitor(position_monitor)

            task.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(),
                                            frame_P_goal=root_P_top,
                                            reference_velocity=self.reference_linear_velocity,
                                            weight=self.weight)

            # Add Vector-Align for better alignment to push later
            if (((i == len(root_P_top_chain) - 1 and align_gripper == MoveAroundHingeAlign.LAST)
                 or (align_gripper == MoveAroundHingeAlign.ALL))
                    and self.tip_gripper_axis is not None):
                task.add_vector_goal_constraints(frame_V_current=self.root_V_tip_grasp_axis,
                                                 frame_V_goal=self.root_V_object_rotation_axis,
                                                 reference_velocity=self.reference_angular_velocity,
                                                 weight=self.weight)

            if i == 0:
                task.start_condition = self.start_condition
                task.end_condition = position_monitor.name
            elif i == len(root_P_top_chain) - 1:
                end_con = f'({self.end_condition}) or {position_monitor.name}'
                task.start_condition = self.old_position_monitor.name
                task.end_condition = end_con
            else:
                task.start_condition = self.old_position_monitor.name
                task.end_condition = position_monitor.name

            self.old_position_monitor = position_monitor

        self.observation_expression = self.old_position_monitor.observation_expression
