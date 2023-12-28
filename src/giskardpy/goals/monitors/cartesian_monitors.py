from typing import List, Dict, Optional

from geometry_msgs.msg import PointStamped, QuaternionStamped, PoseStamped, Vector3Stamped

import giskardpy.casadi_wrapper as cas
from giskardpy.goals.monitors.monitors import ExpressionMonitor
from giskardpy.god_map import god_map
from giskardpy.my_types import Derivatives
import giskardpy.utils.tfwrapper as tf
from giskardpy.utils.expression_definition_utils import transform_msg


class PoseReached(ExpressionMonitor):
    def __init__(self,
                 name: str,
                 root_link: str, tip_link: str,
                 goal_pose: PoseStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 position_threshold: float = 0.01,
                 orientation_threshold: float = 0.01,
                 update_pose_on: Optional[List[str]] = None,
                 crucial: bool = True,
                 stay_one: bool = True):
        super().__init__(name, crucial=crucial, stay_one=stay_one)
        root_link = god_map.world.search_for_link_name(root_link, root_group)
        tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        if update_pose_on is None:
            goal_pose = transform_msg(root_link, goal_pose)
            root_T_goal = cas.TransMatrix(goal_pose)
        else:
            goal_frame_id = god_map.world.search_for_link_name(goal_pose.header.frame_id)
            goal_frame_id_T_goal = transform_msg(goal_frame_id, goal_pose)
            root_T_goal_frame_id = god_map.world.compose_fk_expression(root_link, goal_frame_id)
            root_T_goal_frame_id = god_map.monitor_manager.register_expression_updater(root_T_goal_frame_id,
                                                                                       tuple(update_pose_on))
            goal_frame_id_T_goal = cas.TransMatrix(goal_frame_id_T_goal)
            root_T_goal = root_T_goal_frame_id.dot(goal_frame_id_T_goal)

        # %% position error
        r_P_g = root_T_goal.to_position()
        r_P_c = god_map.world.compose_fk_expression(root_link, tip_link).to_position()
        distance_to_goal = cas.euclidean_distance(r_P_g, r_P_c)
        position_reached = cas.less(distance_to_goal, position_threshold)

        # %% orientation error
        r_R_g = root_T_goal.to_rotation()
        r_R_c = god_map.world.compose_fk_expression(root_link, tip_link).to_rotation()
        rotation_error = cas.rotational_error(r_R_c, r_R_g)
        orientation_reached = cas.less(cas.abs(rotation_error), orientation_threshold)

        self.set_expression(cas.logic_and(position_reached, orientation_reached))


class PositionReached(ExpressionMonitor):
    def __init__(self,
                 name: str,
                 root_link: str, tip_link: str,
                 goal_point: PointStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 threshold: float = 0.01,
                 crucial: bool = True,
                 stay_one: bool = True):
        super().__init__(name, crucial=crucial, stay_one=stay_one)
        root_link = god_map.world.search_for_link_name(root_link, root_group)
        tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        goal_point = transform_msg(root_link, goal_point)
        r_P_g = cas.Point3(goal_point)
        r_P_c = god_map.world.compose_fk_expression(root_link, tip_link).to_position()
        distance_to_goal = cas.euclidean_distance(r_P_g, r_P_c)
        self.set_expression(cas.less(distance_to_goal, threshold))


class OrientationReached(ExpressionMonitor):
    def __init__(self,
                 name: str,
                 root_link: str,
                 tip_link: str,
                 goal_orientation: QuaternionStamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 threshold: float = 0.01,
                 crucial: bool = True,
                 stay_one: bool = True):
        super().__init__(name, crucial=crucial, stay_one=stay_one)
        root_link = god_map.world.search_for_link_name(root_link, root_group)
        tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        goal_orientation = transform_msg(root_link, goal_orientation)
        r_R_g = cas.RotationMatrix(goal_orientation)
        r_R_c = god_map.world.compose_fk_expression(root_link, tip_link).to_rotation()
        rotation_error = cas.rotational_error(r_R_c, r_R_g)
        self.set_expression(cas.less(cas.abs(rotation_error), threshold))


class PointingAt(ExpressionMonitor):
    def __init__(self,
                 name: str,
                 tip_link: str,
                 goal_point: PointStamped,
                 root_link: str,
                 tip_group: Optional[str] = None,
                 root_group: Optional[str] = None,
                 pointing_axis: Vector3Stamped = None,
                 threshold: float = 0.01,
                 crucial: bool = True,
                 stay_one: bool = True):
        super().__init__(name, crucial=crucial, stay_one=stay_one)
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)
        self.root_P_goal_point = transform_msg(self.root, goal_point)

        tip_V_pointing_axis = transform_msg(self.tip, pointing_axis)
        tip_V_pointing_axis.vector = tf.normalize(tip_V_pointing_axis.vector)
        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_P_tip = root_T_tip.to_position()
        root_P_goal_point = cas.Point3(self.root_P_goal_point)
        tip_V_pointing_axis = cas.Vector3(tip_V_pointing_axis)

        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip
        distance = cas.distance_point_to_line(frame_P_point=root_P_goal_point,
                                              frame_P_line_point=root_P_tip,
                                              frame_V_line_direction=root_V_pointing_axis)
        expr = cas.less(cas.abs(distance), threshold)
        self.set_expression(expr)
        god_map.debug_expression_manager.add_debug_expression('point', root_P_goal_point)
        god_map.debug_expression_manager.add_debug_expression('pointing', root_V_pointing_axis)


class VectorsAligned(ExpressionMonitor):
    def __init__(self,
                 name: str,
                 root_link: str,
                 tip_link: str,
                 goal_normal: Vector3Stamped,
                 tip_normal: Vector3Stamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 threshold: float = 0.01,
                 crucial: bool = True,
                 stay_one: bool = True):
        super().__init__(name, crucial=crucial, stay_one=stay_one)
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)

        self.tip_V_tip_normal = transform_msg(self.tip, tip_normal)
        self.tip_V_tip_normal.vector = tf.normalize(self.tip_V_tip_normal.vector)

        self.root_V_root_normal = transform_msg(self.root, goal_normal)
        self.root_V_root_normal.vector = tf.normalize(self.root_V_root_normal.vector)

        tip_V_tip_normal = cas.Vector3(self.tip_V_tip_normal)
        root_R_tip = god_map.world.compose_fk_expression(self.root, self.tip).to_rotation()
        root_V_tip_normal = root_R_tip.dot(tip_V_tip_normal)
        root_V_root_normal = cas.Vector3(self.root_V_root_normal)
        error = cas.angle_between_vector(root_V_tip_normal, root_V_root_normal)
        expr = cas.less(error, threshold)
        self.set_expression(expr)


class DistanceToLine(ExpressionMonitor):
    def __init__(self,
                 name: str,
                 root_link: str,
                 tip_link: str,
                 center_point: PointStamped,
                 line_axis: Vector3Stamped,
                 line_length: float,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 threshold: float = 0.01,
                 crucial: bool = True,
                 stay_one: bool = True):
        super().__init__(name, crucial=crucial, stay_one=stay_one)
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)

        root_P_current = god_map.world.compose_fk_expression(self.root, self.tip).to_position()
        root_V_line_axis = transform_msg(self.root, line_axis)
        root_V_line_axis = cas.Vector3(root_V_line_axis)
        root_V_line_axis.scale(1)
        root_P_center = transform_msg(self.root, center_point)
        root_P_center = cas.Point3(root_P_center)
        root_P_line_start = root_P_center + root_V_line_axis * (line_length / 2)
        root_P_line_end = root_P_center - root_V_line_axis * (line_length / 2)

        distance, closest_point = cas.distance_point_to_line_segment(frame_P_current=root_P_current,
                                                                     frame_P_line_start=root_P_line_start,
                                                                     frame_P_line_end=root_P_line_end)
        expr = cas.less(distance, threshold)
        self.set_expression(expr)
