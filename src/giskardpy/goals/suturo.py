import os
from copy import deepcopy
from enum import Enum
from typing import Optional, Dict

import actionlib
import numpy as np
import rospy
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3, Vector3Stamped, QuaternionStamped, Quaternion

from giskardpy.god_map import god_map
from giskardpy.utils.expression_definition_utils import transform_msg, transform_msg_and_turn_to_expr

if 'GITHUB_WORKFLOW' not in os.environ:
    from tmc_control_msgs.msg import GripperApplyEffortGoal, GripperApplyEffortAction
from trajectory_msgs.msg import JointTrajectoryPoint

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.goals.align_planes import AlignPlanes
from giskardpy.goals.cartesian_goals import CartesianPosition, CartesianOrientation
from giskardpy.goals.goal import Goal, ForceSensorGoal, NonMotionGoal
from giskardpy.tasks.task import WEIGHT_ABOVE_CA
from giskardpy.goals.joint_goals import JointPositionList
from giskardpy.model.links import BoxGeometry, LinkGeometry, SphereGeometry, CylinderGeometry
from giskardpy.utils.logging import loginfo, logwarn

if 'GITHUB_WORKFLOW' not in os.environ:
    from manipulation_msgs.msg import ContextAction, ContextFromAbove, ContextNeatly, ContextObjectType, \
        ContextObjectShape, \
        ContextAlignVertical


class ContextTypes(Enum):
    context_action = ContextAction
    context_from_above = ContextFromAbove
    context_neatly = ContextNeatly
    context_object_type = ContextObjectType
    context_object_shape = ContextObjectShape


class ContextActionModes(Enum):
    grasping = 'grasping'
    placing = 'placing'
    pouring = 'pouring'
    door_opening = 'door-opening'


class ObjectGoal(Goal):
    """
    Inherit from this class if the goal tries to get the object by name from the world
    """

    def get_object_by_name(self, object_name):
        try:
            loginfo('trying to get objects with name')

            object_link = god_map.world.get_link(object_name)
            # TODO: When object has no collision: set size to 0, 0, 0
            object_collisions = object_link.collisions
            if len(object_collisions) == 0:
                object_geometry = BoxGeometry(link_T_geometry=np.eye(4), depth=0, width=0, height=0, color=None)
            else:
                object_geometry: LinkGeometry = object_link.collisions[0]

            goal_pose = god_map.world.compute_fk_pose('map', object_name)

            loginfo(f'goal_pose by name: {goal_pose}')

            # Declare instance of geometry
            if isinstance(object_geometry, BoxGeometry):
                object_type = 'box'
                object_geometry: BoxGeometry = object_geometry
                # FIXME use expression instead of vector3, unless its really a vector
                object_size = Vector3(object_geometry.width, object_geometry.depth, object_geometry.height)

            elif isinstance(object_geometry, CylinderGeometry):
                object_type = 'cylinder'
                object_geometry: CylinderGeometry = object_geometry
                object_size = Vector3(object_geometry.radius, object_geometry.radius, object_geometry.height)

            elif isinstance(object_geometry, SphereGeometry):
                object_type = 'sphere'
                object_geometry: SphereGeometry = object_geometry
                object_size = Vector3(object_geometry.radius, object_geometry.radius, object_geometry.radius)

            else:
                raise Exception('Not supported geometry')

            loginfo(f'Got geometry: {object_type}')
            return goal_pose, object_size

        except:
            loginfo('Could not get geometry from name')
            return None


# TODO move to PayloadMonitor
class MoveGripper(NonMotionGoal):
    _gripper_apply_force_client = actionlib.SimpleActionClient('/hsrb/gripper_controller/grasp',
                                                               GripperApplyEffortAction)
    _gripper_controller = actionlib.SimpleActionClient('/hsrb/gripper_controller/follow_joint_trajectory',
                                                       FollowJointTrajectoryAction)

    def __init__(self,
                 gripper_state: str,
                 suffix=''):
        """
        Open / CLose Gripper.
        Current implementation is only a workaround for manipulation to work with the gripper
        and a follow joint, trajectory connection was not helpful.
        For whole plans, Planning should open the gripper by themselves

        :param gripper_state: keyword to state the gripper. Possible options: 'open', 'neutral', 'close'
        """
        super().__init__()

        self.suffix = suffix
        self.gripper_state = gripper_state

        if self.gripper_state == 'open':
            self.close_gripper_force(0.8)

        elif self.gripper_state == 'close':
            self.close_gripper_force(-0.8)

        elif self.gripper_state == 'neutral':
            self.set_gripper_joint_position(0.5)

    def close_gripper_force(self, force=0.8):
        """
        Closes the gripper with the given force.
        :param force: force to grasp with should be between 0.2 and 0.8 (N)
        :return: applied effort
        """
        rospy.loginfo("Closing gripper with force: {}".format(force))
        f = force  # max(min(0.8, force), 0.2)
        goal = GripperApplyEffortGoal()
        goal.effort = f
        self._gripper_apply_force_client.send_goal(goal)

    def set_gripper_joint_position(self, position):
        """
        Sets the gripper joint to the given  position
        :param position: goal position of the joint -0.105 to 1.239 rad
        :return: error_code of FollowJointTrajectoryResult
        """
        pos = max(min(1.239, position), -0.105)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = [u'hand_motor_joint']
        p = JointTrajectoryPoint()
        p.positions = [pos]
        p.velocities = [0]
        p.effort = [0.1]
        p.time_from_start = rospy.Time(1)
        goal.trajectory.points = [p]
        self._gripper_controller.send_goal(goal)


class Reaching(ObjectGoal):
    def __init__(self,
                 context: {str: ContextTypes},
                 name: str = None,
                 object_name: Optional[str] = None,
                 object_shape: Optional[str] = None,
                 goal_pose: Optional[PoseStamped] = None,
                 object_size: Optional[Vector3] = None,
                 root_link: Optional[str] = None,
                 tip_link: Optional[str] = None,
                 velocity: float = 0.2,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol, ):
        """
            Concludes Reaching type goals.
            Executes them depending on the given context action.
            Context is a dictionary in an action is given as well as situational parameters.
            All available context Messages are found in the Enum 'ContextTypes'

            :param context: Context of this goal. Contains information about action and situational parameters
            :param object_name: Name of the object to use. Optional as long as goal_pose and object_size are filled instead
            :param object_shape: Shape of the object to manipulate. Edit object size when having a sphere or cylinder
            :param goal_pose: Goal pose for the object. Alternative if no object name is given.
            :param object_size: Given object size. Alternative if no object name is given.
            :param root_link: Current root Link
            :param tip_link: Current tip link
            :param velocity: Desired velocity of this goal
            :param weight: weight of this goal
        """
        if name is None:
            name = 'Reaching'

        super().__init__(name)

        if root_link is None:
            root_link = god_map.world.groups[god_map.world.robot_name].root_link.name.short_name
        if tip_link is None:
            tip_link = self.gripper_tool_frame

        self.context = context
        self.object_name = object_name
        self.object_shape = object_shape
        self.root_link_name = root_link
        self.tip_link_name = tip_link
        self.velocity = velocity
        self.weight = weight
        self.action = check_context_element('action', ContextAction, self.context)
        self.from_above = check_context_element('from_above', ContextFromAbove, self.context)
        self.align_vertical = check_context_element('align_vertical', ContextAlignVertical, self.context)
        self.radius = 0.0
        self.careful = False
        self.object_in_world = goal_pose is None

        # Get object geometry from name
        if goal_pose is None:
            self.goal_pose, self.object_size = self.get_object_by_name(self.object_name)
            self.reference_frame = self.object_name

        else:
            try:
                god_map.world.search_for_link_name(goal_pose.header.frame_id)
                self.goal_pose = goal_pose
            except:
                logwarn(f'Couldn\'t find {goal_pose.header.frame_id}. Searching in tf.')
                self.goal_pose = tf.lookup_pose('map', goal_pose)

            self.object_size = object_size
            self.reference_frame = 'base_footprint'
            logwarn(f'Warning: Object not in giskard world')

        if self.action == ContextActionModes.grasping.value:
            if self.object_shape == 'sphere' or self.object_shape == 'cylinder':
                self.radius = self.object_size.x

            elif self.object_name == 'plate':
                self.radius = -(self.object_size.x / 2) + 0.03

            elif self.object_name == 'bowl':
                print('Bowl!')
                object_size = Vector3(0.16, 0.16, 0.058)
                self.radius = -(object_size.x / 2) + 0.15

            else:
                if self.from_above:
                    pass
                elif self.object_in_world:
                    self.radius = - 0.02
                else:
                    self.radius = max(min(0.08, self.object_size.x / 2), 0.05)

        elif self.action == ContextActionModes.placing.value:
            if self.object_shape == 'sphere' or self.object_shape == 'cylinder':
                self.radius = self.object_size.x

            # Placing positions are calculated in planning in clean the table.
            # Apply height offset only when placing frontal
            if not self.from_above:
                self.goal_pose.pose.position.z += (self.object_size.z / 2) + 0.02

        elif self.action == ContextActionModes.pouring.value:
            # Pouring position is calculated in planning in serve breakfast.
            pass

        elif self.action == ContextActionModes.door_opening.value:
            self.radius = -0.02
            self.goal_pose = transform_msg(god_map.world.search_for_link_name('base_footprint'), self.goal_pose)
            self.careful = True

        if self.careful:
            self.add_constraints_of_goal(GraspCarefully(goal_pose=self.goal_pose,
                                                        reference_frame_alignment=self.reference_frame,
                                                        frontal_offset=self.radius,
                                                        from_above=self.from_above,
                                                        align_vertical=self.align_vertical,
                                                        root_link=self.root_link_name,
                                                        tip_link=self.tip_link_name,
                                                        velocity=self.velocity / 2,
                                                        weight=self.weight,
                                                        start_condition=start_condition,
                                                        hold_condition=hold_condition,
                                                        end_condition=end_condition))
        else:
            self.add_constraints_of_goal(GraspObject(goal_pose=self.goal_pose,
                                                     reference_frame_alignment=self.reference_frame,
                                                     frontal_offset=self.radius,
                                                     from_above=self.from_above,
                                                     align_vertical=self.align_vertical,
                                                     root_link=self.root_link_name,
                                                     tip_link=self.tip_link_name,
                                                     velocity=self.velocity,
                                                     weight=self.weight,
                                                     start_condition=start_condition,
                                                     hold_condition=hold_condition,
                                                     end_condition=end_condition))


class GraspObject(ObjectGoal):
    def __init__(self,
                 goal_pose: PoseStamped,
                 frontal_offset: float = 0.0,
                 from_above: bool = False,
                 align_vertical: bool = False,
                 name: Optional[str] = None,
                 reference_frame_alignment: Optional[str] = None,
                 root_link: Optional[str] = None,
                 tip_link: Optional[str] = None,
                 velocity: float = 0.2,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):
        """
            Concludes Reaching type goals.
            Executes them depending on the given context action.
            Context is a dictionary in an action is given as well as situational parameters.
            All available context Messages are found in the Enum 'ContextTypes'

            :param goal_pose: Goal pose for the object.
            :param frontal_offset: Optional parameter to pass a specific offset
            :param from_above: States if the gripper should be aligned frontal or from above
            :param align_vertical: States if the gripper should be rotated.
            :param reference_frame_alignment: Reference frame to align with. Is usually either an object link or 'base_footprint'
            :param root_link: Current root Link
            :param tip_link: Current tip link
            :param velocity: Desired velocity of this goal
            :param weight: weight of this goal
        """
        if name is None:
            name = 'GraspObject'

        super().__init__(name=name)
        self.goal_pose = goal_pose

        self.frontal_offset = frontal_offset
        self.from_above = from_above
        self.align_vertical = align_vertical

        if reference_frame_alignment is None:
            reference_frame_alignment = 'base_footprint'

        if root_link is None:
            root_link = god_map.world.groups[god_map.world.robot_name].root_link.name

        if tip_link is None:
            tip_link = self.gripper_tool_frame

        self.reference_link = god_map.world.search_for_link_name(reference_frame_alignment)
        self.root_link = god_map.world.search_for_link_name(root_link)
        self.tip_link = god_map.world.search_for_link_name(tip_link)

        self.velocity = velocity
        self.weight = weight

        self.goal_frontal_axis = Vector3Stamped()
        self.goal_frontal_axis.header.frame_id = self.reference_link.short_name

        self.tip_frontal_axis = Vector3Stamped()
        self.tip_frontal_axis.header.frame_id = self.tip_link.short_name

        self.goal_vertical_axis = Vector3Stamped()
        self.goal_vertical_axis.header.frame_id = self.reference_link.short_name

        self.tip_vertical_axis = Vector3Stamped()
        self.tip_vertical_axis.header.frame_id = self.tip_link.short_name

        root_goal_point = PointStamped()
        root_goal_point.header.frame_id = self.goal_pose.header.frame_id
        root_goal_point.point = self.goal_pose.pose.position

        self.goal_point = transform_msg(self.reference_link, root_goal_point)

        if self.from_above:
            self.goal_vertical_axis.vector = self.standard_forward
            self.goal_frontal_axis.vector = multiply_vector(self.standard_up, -1)
            self.goal_point.point.y += frontal_offset
            if frontal_offset > 0:
                self.goal_point.point.z += 0.04

        else:
            self.goal_vertical_axis.vector = self.standard_up
            self.goal_frontal_axis.vector = self.base_forward

            self.goal_point.point.x += frontal_offset
            self.goal_point.point.z -= 0.01

        if self.align_vertical:
            self.tip_vertical_axis.vector = self.gripper_left

        else:
            self.tip_vertical_axis.vector = self.gripper_up

        self.tip_frontal_axis.vector = self.gripper_forward

        # Position
        self.add_constraints_of_goal(CartesianPosition(root_link=self.root_link.short_name,
                                                       tip_link=self.tip_link.short_name,
                                                       goal_point=self.goal_point,
                                                       reference_velocity=self.velocity,
                                                       weight=self.weight,
                                                       start_condition=start_condition,
                                                       hold_condition=hold_condition,
                                                       end_condition=end_condition))

        # FIXME you can use orientation goal instead of two align planes
        # Align vertical
        self.add_constraints_of_goal(AlignPlanes(root_link=self.root_link.short_name,
                                                 tip_link=self.tip_link.short_name,
                                                 goal_normal=self.goal_vertical_axis,
                                                 tip_normal=self.tip_vertical_axis,
                                                 reference_velocity=self.velocity,
                                                 weight=self.weight,
                                                 start_condition=start_condition,
                                                 hold_condition=hold_condition,
                                                 end_condition=end_condition))

        # Align frontal
        self.add_constraints_of_goal(AlignPlanes(root_link=self.root_link.short_name,
                                                 tip_link=self.tip_link.short_name,
                                                 goal_normal=self.goal_frontal_axis,
                                                 tip_normal=self.tip_frontal_axis,
                                                 reference_velocity=self.velocity,
                                                 weight=self.weight,
                                                 start_condition=start_condition,
                                                 hold_condition=hold_condition,
                                                 end_condition=end_condition))

        self.add_constraints_of_goal(KeepRotationGoal(tip_link='base_footprint',
                                                      weight=self.weight,
                                                      start_condition=start_condition,
                                                      hold_condition=hold_condition,
                                                      end_condition=end_condition))

        # god_map.debug_expression_manager.add_debug_expression('goal point', self.goal_point)


class VerticalMotion(ObjectGoal):
    def __init__(self,
                 context: {str: ContextTypes},
                 name: str = None,
                 distance: float = 0.02,
                 root_link: Optional[str] = None,
                 tip_link: Optional[str] = None,
                 velocity: float = 0.2,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):
        """
        Move the tip link vertical according to the given context.

        :param context: Same parameter as in the goal 'Reaching'
        :param distance: Optional parameter to adjust the distance to move.
        :param root_link: Current root Link
        :param tip_link: Current tip link
        :param velocity: Desired velocity of this goal
        :param weight: weight of this goal
        """
        if name is None:
            name = 'VerticalMotion'

        super().__init__(name)

        if root_link is None:
            root_link = 'base_footprint'
        if tip_link is None:
            tip_link = self.gripper_tool_frame
        self.context = context
        self.distance = distance
        self.root_link = god_map.world.search_for_link_name(root_link)
        self.tip_link = god_map.world.search_for_link_name(tip_link)
        self.velocity = velocity
        self.weight = weight
        self.base_footprint = god_map.world.search_for_link_name('base_footprint')
        self.action = check_context_element('action', ContextAction, self.context)

        start_point_tip = PoseStamped()
        start_point_tip.header.frame_id = self.tip_link.short_name
        goal_point_base = transform_msg(self.base_footprint, start_point_tip)

        up = ContextActionModes.grasping.value in self.action
        down = ContextActionModes.placing.value in self.action
        if up:
            goal_point_base.pose.position.z += self.distance
        elif down:
            goal_point_base.pose.position.z -= self.distance
        else:
            logwarn('no direction given')

        self.add_constraints_of_goal(KeepRotationGoal(tip_link=self.tip_link.short_name,
                                                      weight=self.weight,
                                                      start_condition=start_condition,
                                                      hold_condition=hold_condition,
                                                      end_condition=end_condition))

        goal_point_tip = transform_msg(self.tip_link, goal_point_base)
        self.goal_point = deepcopy(goal_point_tip)
        # self.root_T_tip_start = god_map.world.compute_fk_np(self.root_link, self.tip_link)
        # self.start_tip_T_current_tip = np.eye(4)

        # start_tip_T_current_tip = w.TransMatrix(self.get_parameter_as_symbolic_expression('start_tip_T_current_tip'))
        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)

        # t_T_g = w.TransMatrix(self.goal_point)
        # r_T_tip_eval = w.TransMatrix(god_map.evaluate_expr(root_T_tip))

        # root_T_goal = r_T_tip_eval.dot(start_tip_T_current_tip).dot(t_T_g)

        root_T_goal = transform_msg_and_turn_to_expr(self.root_link, self.goal_point, condition=start_condition)

        r_P_g = root_T_goal.to_position()
        r_P_c = root_T_tip.to_position()

        task = self.create_and_add_task(task_name='VerticalMotion')

        task.add_point_goal_constraints(frame_P_goal=r_P_g,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.velocity,
                                        weight=self.weight)

        self.connect_monitors_to_all_tasks(start_condition=start_condition, hold_condition=hold_condition,
                                           end_condition=end_condition)


class Retracting(ObjectGoal):
    def __init__(self,
                 object_name='',
                 name: str = None,
                 distance: float = 0.3,
                 reference_frame: Optional[str] = None,
                 root_link: Optional[str] = None,
                 tip_link: Optional[str] = None,
                 velocity: float = 0.2,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):
        """
        Retract the tip link from the current position by the given distance.
        The exact direction is based on the given reference frame.

        :param object_name: Unused parameter that exists because cram throws errors when calling a goal without a parameter
        :param distance: Optional parameter to adjust the distance to move.
        :param reference_frame: Reference axis from which should be retracted. Is usually 'base_footprint' or 'hand_palm_link'
        :param root_link: Current root Link
        :param tip_link: Current tip link
        :param velocity: Desired velocity of this goal
        :param weight: weight of this goal

        """
        if name is None:
            name = 'Retracting'

        super().__init__(name)

        if reference_frame is None:
            reference_frame = 'base_footprint'
        if root_link is None:
            root_link = god_map.world.groups[god_map.world.robot_name].root_link.name
        if tip_link is None:
            tip_link = self.gripper_tool_frame
        self.distance = distance
        self.reference_frame = god_map.world.search_for_link_name(reference_frame)
        self.root_link = god_map.world.search_for_link_name(root_link)
        self.tip_link = god_map.world.search_for_link_name(tip_link)
        self.velocity = velocity
        self.weight = weight
        self.hand_frames = [self.gripper_tool_frame, 'hand_palm_link']

        tip_P_start = PoseStamped()
        tip_P_start.header.frame_id = self.tip_link.short_name
        tip_P_start.pose.orientation.w = 1
        reference_P_start = transform_msg(self.reference_frame, tip_P_start)

        if self.reference_frame.short_name in self.hand_frames:
            reference_P_start.pose.position.z -= self.distance
        else:
            reference_P_start.pose.position.x -= self.distance

        self.goal_point = transform_msg(self.tip_link, reference_P_start)
        # self.root_T_tip_start = god_map.world.compute_fk_np(self.root_link, self.tip_link)
        # self.start_tip_T_current_tip = np.eye(4)
        self.add_constraints_of_goal(KeepRotationGoal(tip_link='base_footprint',
                                                      weight=self.weight,
                                                      start_condition=start_condition,
                                                      hold_condition=hold_condition,
                                                      end_condition=end_condition))

        if 'base' not in self.tip_link.short_name:
            self.add_constraints_of_goal(KeepRotationGoal(tip_link=self.tip_link.short_name,
                                                          weight=self.weight,
                                                          start_condition=start_condition,
                                                          hold_condition=hold_condition,
                                                          end_condition=end_condition))

        task = self.create_and_add_task('Retracting')

        # start_tip_T_current_tip = w.TransMatrix(self.get_parameter_as_symbolic_expression('start_tip_T_current_tip'))
        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)

        # t_T_g = w.TransMatrix(self.goal_point)
        # r_T_tip_eval = w.TransMatrix(god_map.evaluate_expr(root_T_tip))

        # root_T_goal = r_T_tip_eval.dot(start_tip_T_current_tip).dot(t_T_g)

        root_T_goal = transform_msg_and_turn_to_expr(self.root_link, self.goal_point, condition=start_condition)

        r_P_g = root_T_goal.to_position()
        r_P_c = root_T_tip.to_position()

        task.add_point_goal_constraints(frame_P_goal=r_P_g,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.velocity,
                                        weight=self.weight)

        self.connect_monitors_to_all_tasks(start_condition=start_condition, hold_condition=hold_condition,
                                           end_condition=end_condition)


class AlignHeight(ObjectGoal):
    def __init__(self,
                 context: {str: ContextTypes},
                 name: str = None,
                 object_name: Optional[str] = None,
                 goal_pose: Optional[PoseStamped] = None,
                 object_height: float = 0.0,
                 root_link: Optional[str] = None,
                 tip_link: Optional[str] = None,
                 velocity: float = 0.2,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):
        """
        Align the tip link with the given goal_pose to prepare for further action (e.g. grasping or placing)

        :param context: Same parameter as in the goal 'Reaching'
        :param object_name: name of the object if added to world
        :param goal_pose: final destination pose
        :param object_height: height of the target object. Used as additional offset.
        :param root_link: Current root Link
        :param tip_link: Current tip link
        :param velocity: Desired velocity of this goal
        :param weight: weight of this goal
        """
        if name is None:
            name = 'AlignHeight'

        super().__init__(name)

        self.object_name = object_name

        # Get object from name
        if goal_pose is None:
            goal_pose, object_size = self.get_object_by_name(self.object_name)

            object_height = object_size.z

        try:
            god_map.world.search_for_link_name(goal_pose.header.frame_id)
            self.goal_pose = goal_pose
        except:
            logwarn(f'Couldn\'t find {goal_pose.header.frame_id}. Searching in tf.')
            self.goal_pose = tf.lookup_pose('map', goal_pose)

        self.object_height = object_height

        if root_link is None:
            root_link = god_map.world.groups[god_map.world.robot_name].root_link.name
        if tip_link is None:
            tip_link = self.gripper_tool_frame

        self.root_link = god_map.world.search_for_link_name(root_link)
        self.tip_link = god_map.world.search_for_link_name(tip_link)

        self.velocity = velocity
        self.weight = weight

        self.from_above = check_context_element('from_above', ContextFromAbove, context)

        self.base_footprint = god_map.world.search_for_link_name('base_footprint')

        goal_point = PointStamped()
        goal_point.header.frame_id = self.goal_pose.header.frame_id
        goal_point.point = self.goal_pose.pose.position

        base_to_tip = god_map.world.compute_fk_pose(self.base_footprint, self.tip_link)

        offset = 0.02
        base_goal_point = transform_msg(self.base_footprint, goal_point)
        base_goal_point.point.x = base_to_tip.pose.position.x
        base_goal_point.point.z += (self.object_height / 2) + offset

        if self.from_above:
            # Tip facing downwards
            base_goal_point.point.z += 0.05

            base_V_g = Vector3Stamped()
            base_V_g.header.frame_id = self.base_footprint.short_name
            base_V_g.vector.z = -1

            tip_V_g = Vector3Stamped()
            tip_V_g.header.frame_id = self.tip_link.short_name
            tip_V_g.vector = self.gripper_forward

            base_V_x = Vector3Stamped()
            base_V_x.header.frame_id = self.base_footprint.short_name
            base_V_x.vector.x = 1

            tip_V_x = Vector3Stamped()
            tip_V_x.header.frame_id = self.tip_link.short_name
            tip_V_x.vector.x = 1

            self.add_constraints_of_goal(AlignPlanes(root_link=self.root_link.short_name,
                                                     tip_link=self.tip_link.short_name,
                                                     goal_normal=base_V_g,
                                                     tip_normal=tip_V_g))

            self.add_constraints_of_goal(AlignPlanes(root_link=self.root_link.short_name,
                                                     tip_link=self.tip_link.short_name,
                                                     goal_normal=base_V_x,
                                                     tip_normal=tip_V_x))

        else:
            # Tip facing frontal
            self.add_constraints_of_goal(KeepRotationGoal(tip_link=self.tip_link.short_name,
                                                          weight=self.weight,
                                                          start_condition=start_condition,
                                                          hold_condition=hold_condition,
                                                          end_condition=end_condition))

        self.add_constraints_of_goal(KeepRotationGoal(tip_link=self.base_footprint.short_name,
                                                      weight=self.weight,
                                                      start_condition=start_condition,
                                                      hold_condition=hold_condition,
                                                      end_condition=end_condition))

        self.goal_point = transform_msg(self.tip_link, base_goal_point)

        self.add_constraints_of_goal(CartesianPosition(root_link=self.root_link.short_name,
                                                       tip_link=self.tip_link.short_name,
                                                       goal_point=self.goal_point,
                                                       reference_velocity=self.velocity,
                                                       weight=self.weight,
                                                       start_condition=start_condition,
                                                       hold_condition=hold_condition,
                                                       end_condition=end_condition))


class GraspCarefully(ForceSensorGoal):
    def __init__(self,
                 goal_pose: PoseStamped,
                 name: str = None,
                 frontal_offset: float = 0.0,
                 from_above: bool = False,
                 align_vertical: bool = False,
                 reference_frame_alignment: Optional[str] = None,
                 root_link: Optional[str] = None,
                 tip_link: Optional[str] = None,
                 velocity: float = 0.02,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):
        """
        Same as GraspObject but with force sensor to avoid bumping into things (e.g. door for door opening).

        :param goal_pose: Goal pose for the object.
        :param frontal_offset: Optional parameter to pass a specific offset
        :param from_above: States if the gripper should be aligned frontal or from above
        :param align_vertical: States if the gripper should be rotated.
        :param reference_frame_alignment: Reference frame to align with. Is usually either an object link or 'base_footprint'
        :param root_link: Current root Link
        :param tip_link: Current tip link
        :param velocity: Desired velocity of this goal
        :param weight: weight of this goal

        """
        if name is None:
            name = 'GraspCarefully'

        # FIXME: ForceSensorGoal muss name annehmen, dann name als Parameter übergeben
        super().__init__()

        self.add_constraints_of_goal(GraspObject(goal_pose=goal_pose,
                                                 reference_frame_alignment=reference_frame_alignment,
                                                 frontal_offset=frontal_offset,
                                                 from_above=from_above,
                                                 align_vertical=align_vertical,
                                                 root_link=root_link,
                                                 tip_link=tip_link,
                                                 velocity=velocity,
                                                 weight=weight,
                                                 start_condition=start_condition,
                                                 hold_condition=hold_condition,
                                                 end_condition=end_condition))

    # might need to be removed in the future, as soon as the old interface isn't used by anymore
    def goal_cancel_condition(self):
        force_threshold = 5.0

        expression = (lambda sensor_values, _:
                      (abs(sensor_values['x_force']) >= force_threshold) or
                      (abs(sensor_values['y_force']) >= force_threshold) or
                      (abs(sensor_values['z_force']) >= force_threshold))

        return expression

    def recovery(self) -> Dict:
        return {}


class Placing(ObjectGoal):

    def __init__(self,
                 context: {str: ContextTypes},
                 goal_pose: PoseStamped,
                 name: str = None,
                 root_link: Optional[str] = None,
                 tip_link: Optional[str] = None,
                 velocity: float = 0.02,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):

        """
        Place an object using the force-/torque-sensor.

        :param context: Context similar to 'Reaching'. Only uses 'from_above' and 'align_vertical' as variables
        :param goal_pose: Goal pose for the object.
        :param root_link: Current root Link
        :param tip_link: Current tip link
        :param velocity: Desired velocity of this goal
        :param weight: weight of this goal
        """
        if name is None:
            name = 'Placing'

        self.goal_pose = goal_pose
        self.velocity = velocity
        self.weight = weight

        self.from_above = check_context_element('from_above', ContextFromAbove, context)
        self.align_vertical = check_context_element('align_vertical', ContextAlignVertical, context)

        # FIXME Wenn ForceSensorGoal name hat, dann muss hier name eingefügt werden
        super().__init__(name=name)

        if root_link is None:
            root_link = 'base_footprint'

        if tip_link is None:
            tip_link = self.gripper_tool_frame

        self.root_link = god_map.world.search_for_link_name(root_link)
        self.tip_link = god_map.world.search_for_link_name(tip_link)

        self.add_constraints_of_goal(GraspObject(goal_pose=self.goal_pose,
                                                 from_above=self.from_above,
                                                 align_vertical=self.align_vertical,
                                                 root_link=self.root_link.short_name,
                                                 tip_link=self.tip_link.short_name,
                                                 velocity=self.velocity,
                                                 weight=self.weight,
                                                 start_condition=start_condition,
                                                 hold_condition=hold_condition,
                                                 end_condition=end_condition))

    # might need to be removed in the future, as soon as the old interface isn't in use anymore
    def goal_cancel_condition(self):

        if self.from_above:

            y_torque_threshold = -0.15

            z_force_threshold = 1.0

            expression = (lambda sensor_values, _:
                          (sensor_values[self.forward_force] >= z_force_threshold))
            # or
            # ((sensor_values[self.sideway_torque] >= y_torque_threshold))

        else:
            x_force_threshold = 0.0
            y_torque_threshold = 0.15

            expression = (lambda sensor_values, _:
                          (sensor_values[self.upwards_force] <= x_force_threshold) or
                          (sensor_values[self.sideway_torque] >= y_torque_threshold))

        return expression

    def recovery_modifier(self) -> Dict:
        joint_states = {'arm_lift_joint': 0.03}

        return joint_states


class Tilting(Goal):
    def __init__(self,
                 name: str = None,
                 direction: Optional[str] = None,
                 angle: Optional[float] = None,
                 tip_link: str = 'wrist_flex_joint',
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):
        """
        Tilts the given tip link into one direction by a given angle.

        :param direction: Direction in which to rotate the joint.
        :param angle: Amount how much the joint should be moved
        :param tip_link: The joint that should rotate. Default ensures correct usage for pouring.

        """
        if name is None:
            name = 'Tilting'
        super().__init__(name)

        max_angle = -2.0

        if angle is None:
            angle = max_angle

        if direction == 'right':
            angle = abs(angle)
        else:
            angle = abs(angle) * -1

        wrist_state = angle
        self.tip_link = tip_link

        self.goal_state = {self.tip_link: wrist_state}

        self.add_constraints_of_goal(JointPositionList(goal_state=self.goal_state,
                                                       start_condition=start_condition,
                                                       hold_condition=hold_condition,
                                                       end_condition=end_condition))


class TakePose(Goal):
    def __init__(self,
                 pose_keyword: str,
                 name: str = None,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):
        """
        Get into a predefined pose with a given keyword.
        Used to get into complete poses. To move only specific joints use 'JointPositionList'

        :param pose_keyword: Keyword for the given poses
        """
        if name is None:
            name = f'TakePose-{pose_keyword}'
        super().__init__(name)

        if pose_keyword == 'park':
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.0
            arm_flex_joint = 0.0
            arm_roll_joint = -1.5
            wrist_flex_joint = -1.5
            wrist_roll_joint = 0.0

        elif pose_keyword == 'perceive':
            head_pan_joint = 0.0
            head_tilt_joint = -0.65
            arm_lift_joint = 0.25
            arm_flex_joint = 0.0
            arm_roll_joint = 1.5
            wrist_flex_joint = -1.5
            wrist_roll_joint = 0.0

        elif pose_keyword == 'assistance':
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.0
            arm_flex_joint = 0.0
            arm_roll_joint = -1.5
            wrist_flex_joint = -1.5
            wrist_roll_joint = 1.6

        elif pose_keyword == 'pre_align_height':
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.0
            arm_flex_joint = 0.0
            arm_roll_joint = 0.0
            wrist_flex_joint = -1.5
            wrist_roll_joint = 0.0

        elif pose_keyword == 'carry':
            head_pan_joint = 0.0
            head_tilt_joint = -0.65
            arm_lift_joint = 0.0
            arm_flex_joint = -0.43
            arm_roll_joint = 0.0
            wrist_flex_joint = -1.17
            wrist_roll_joint = -1.62

        elif pose_keyword == 'test':
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.38
            arm_flex_joint = -1.44
            arm_roll_joint = 0.0
            wrist_flex_joint = -0.19
            wrist_roll_joint = 0.0

        else:
            loginfo(f'{pose_keyword} is not a valid pose')
            return

        joint_states = {
            'head_pan_joint': head_pan_joint,
            'head_tilt_joint': head_tilt_joint,
            'arm_lift_joint': arm_lift_joint,
            'arm_flex_joint': arm_flex_joint,
            'arm_roll_joint': arm_roll_joint,
            'wrist_flex_joint': wrist_flex_joint,
            'wrist_roll_joint': wrist_roll_joint}
        self.goal_state = joint_states

        self.add_constraints_of_goal(JointPositionList(goal_state=self.goal_state,
                                                       start_condition=start_condition,
                                                       hold_condition=hold_condition,
                                                       end_condition=end_condition))


class Mixing(Goal):
    def __init__(self,
                 name=None,
                 mixing_time: float = 20,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):
        """
        Simple Mixing motion.

        :param mixing_time: States how long this goal should be executed.
        :param weight: weight of this goal
        """
        super().__init__()

        self.weight = weight

        target_speed = 1

        self.add_constraints_of_goal(JointRotationGoalContinuous(joint_name='wrist_roll_joint',
                                                                 joint_center=0.0,
                                                                 joint_range=0.9,
                                                                 trajectory_length=mixing_time,
                                                                 target_speed=target_speed,
                                                                 start_condition=start_condition,
                                                                 hold_condition=hold_condition,
                                                                 end_condition=end_condition))

        self.add_constraints_of_goal(JointRotationGoalContinuous(joint_name='wrist_flex_joint',
                                                                 joint_center=-1.3,
                                                                 joint_range=0.2,
                                                                 trajectory_length=mixing_time,
                                                                 target_speed=target_speed,
                                                                 start_condition=start_condition,
                                                                 hold_condition=hold_condition,
                                                                 end_condition=end_condition))

        self.add_constraints_of_goal(JointRotationGoalContinuous(joint_name='arm_roll_joint',
                                                                 joint_center=0.0,
                                                                 joint_range=0.1,
                                                                 trajectory_length=mixing_time,
                                                                 target_speed=target_speed,
                                                                 start_condition=start_condition,
                                                                 hold_condition=hold_condition,
                                                                 end_condition=end_condition))


class JointRotationGoalContinuous(Goal):
    def __init__(self,
                 joint_name: str,
                 joint_center: float,
                 joint_range: float,
                 name: str = None,
                 trajectory_length: float = 20,
                 target_speed: float = 1,
                 period_length: float = 1.0,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):
        """
        Rotate a joint continuously around a center. The execution time and speed is variable.

        :param joint_name: joint name that should be rotated
        :param joint_center: Center of the rotation point
        :param joint_range: Range of the rotational movement. Note that this is calculated + and - joint_center.
        :param trajectory_length: length of this goal in seconds.
        :param target_speed: execution speed of this goal. Adjust when the trajectory is not executed right
        :param period_length: length of the period that should be executed. Adjust when the trajectory is not executed right.
        """
        super().__init__()
        self.joint = god_map.world.search_for_joint_name(joint_name)
        self.target_speed = target_speed
        self.trajectory_length = trajectory_length
        self.joint_center = joint_center
        self.joint_range = joint_range
        self.period_length = period_length

    def make_constraints(self):
        time = self.traj_time_in_seconds()
        joint_position = self.get_joint_position_symbol(self.joint)

        joint_goal = self.joint_center + (w.cos(time * np.pi * self.period_length) * self.joint_range)

        self.add_debug_expr(f'{self.joint.short_name}_goal', joint_goal)
        self.add_debug_expr(f'{self.joint.short_name}_position', joint_position)

        self.add_position_constraint(expr_current=joint_position,
                                     expr_goal=joint_goal,
                                     reference_velocity=self.target_speed,
                                     weight=w.if_greater(time, self.trajectory_length, 0, WEIGHT_ABOVE_CA),
                                     name=self.joint.short_name)


class KeepRotationGoal(Goal):
    def __init__(self,
                 tip_link: str,
                 name: str = None,
                 weight: float = WEIGHT_ABOVE_CA,
                 start_condition: w.Expression = w.TrueSymbol,
                 hold_condition: w.Expression = w.FalseSymbol,
                 end_condition: w.Expression = w.TrueSymbol):
        """
        Use this if a specific link should not rotate during a goal execution. Typically used for the hand.

        :param tip_link: link that shall keep its rotation
        :param weight: weight of this goal
        """
        if name is None:
            name = 'KeepRotationGoal'

        super().__init__(name)

        self.tip_link = tip_link
        self.weight = weight

        zero_quaternion = Quaternion(x=0, y=0, z=0, w=1)
        tip_orientation = QuaternionStamped(quaternion=zero_quaternion)
        tip_orientation.header.frame_id = self.tip_link

        self.add_constraints_of_goal(CartesianOrientation(root_link='map',
                                                          tip_link=self.tip_link,
                                                          goal_orientation=tip_orientation,
                                                          weight=self.weight,
                                                          start_condition=start_condition,
                                                          hold_condition=hold_condition,
                                                          end_condition=end_condition))


def check_context_element(name: str,
                          context_type,
                          context):
    if name in context:
        if isinstance(context[name], context_type):
            return context[name].content
        else:
            return context[name]


def multiply_vector(vec: Vector3,
                    number: int):
    return Vector3(vec.x * number, vec.y * number, vec.z * number)

# TODO: Make Cartesian Orientation from two alignplanes
