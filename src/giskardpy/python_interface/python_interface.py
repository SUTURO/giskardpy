from collections import defaultdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import rospy
from actionlib import SimpleActionClient
from controller_manager_msgs.srv import ListControllers, SwitchController, SwitchControllerResponse, \
    ListControllersResponse
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PointStamped, QuaternionStamped, Vector3, Quaternion
from nav_msgs.msg import Path
from rospy import ServiceException
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_matrix

import giskard_msgs.msg as giskard_msgs
from giskard_msgs.msg import MoveAction, MoveGoal, WorldBody, CollisionEntry, MoveResult, MoveFeedback, MotionGoal, \
    Monitor, WorldGoal, WorldAction, WorldResult, GiskardError
from giskard_msgs.srv import DyeGroupRequest, DyeGroup, GetGroupInfoRequest, DyeGroupResponse
from giskard_msgs.srv import GetGroupInfo, GetGroupNames
from giskard_msgs.srv import GetGroupNamesResponse, GetGroupInfoResponse
from giskardpy.data_types import goal_parameter
from giskardpy.exceptions import DuplicateNameException, UnknownGroupException
from giskardpy.goals.align_planes import AlignPlanes
from giskardpy.goals.align_to_push_door import AlignToPushDoor
from giskardpy.goals.base_traj_follower import CarryMyBullshit, FollowNavPath
from giskardpy.goals.cartesian_goals import CartesianPose, DiffDriveBaseGoal, CartesianVelocityLimit, \
    CartesianOrientation, CartesianPoseStraight, CartesianPosition, CartesianPositionStraight
from giskardpy.goals.collision_avoidance import CollisionAvoidance
from giskardpy.goals.grasp_bar import GraspBar
from giskardpy.goals.joint_goals import JointPositionList, AvoidJointLimits
from giskardpy.goals.open_close import Close, Open
from giskardpy.goals.pointing import Pointing
from giskardpy.goals.pre_push_door import PrePushDoor
from giskardpy.goals.realtime_goals import RealTimePointing
from giskardpy.goals.suturo import GraspBarOffset, Reaching, Placing, VerticalMotion, AlignHeight, TakePose, Tilting, \
    JointRotationGoalContinuous, Mixing, OpenDoorGoal, Retracting, MoveAroundDishwasher
from giskardpy.model.utils import make_world_body_box
from giskardpy.monitors.cartesian_monitors import PoseReached, PositionReached, OrientationReached, PointingAt, \
    VectorsAligned, DistanceToLine
from giskardpy.monitors.force_torque_monitor import PayloadForceTorque
from giskardpy.monitors.joint_monitors import JointGoalReached
from giskardpy.monitors.lidar_monitor import LidarPayloadMonitor
from giskardpy.monitors.monitors import LocalMinimumReached, TimeAbove, Alternator, CancelMotion, EndMotion
from giskardpy.monitors.overwrite_state_monitors import SetOdometry, SetSeedConfiguration
from giskardpy.monitors.payload_monitors import Print, Sleep, SetMaxTrajectoryLength, PayloadAlternator
from giskardpy.monitors.set_prediction_horizon import SetPredictionHorizon
from giskardpy.tasks.task import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.tree.control_modes import ControlModes
from giskardpy.utils.utils import kwargs_to_json, get_all_classes_in_package
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from giskardpy.goals.adaptive_goals import PouringAdaptiveTilt


class WorldWrapper:
    def __init__(self, node_name: str):
        self._get_group_info_srv = rospy.ServiceProxy(f'{node_name}/get_group_info', GetGroupInfo)
        self._get_group_names_srv = rospy.ServiceProxy(f'{node_name}/get_group_names', GetGroupNames)
        self._dye_group_srv = rospy.ServiceProxy(f'{node_name}/dye_group', DyeGroup)
        self._control_mode_srv = rospy.ServiceProxy(f'{node_name}/get_control_mode', Trigger)
        self._client = SimpleActionClient(f'{node_name}/update_world', WorldAction)
        self._client.wait_for_server()
        rospy.wait_for_service(self._get_group_names_srv.resolved_name)
        self.robot_name = self.get_group_names()[0]

    def clear(self) -> WorldResult:
        """
        Resets the world to what it was when Giskard was launched.
        """
        req = WorldGoal()
        req.operation = WorldGoal.REMOVE_ALL
        return self._send_goal_and_wait(req)

    def remove_group(self, name: str) -> WorldResult:
        """
        Removes a group and all links and joints it contains from the world.
        Be careful, you can remove parts of the robot like that.
        """
        world_body = WorldBody()
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.REMOVE
        req.body = world_body
        return self._send_goal_and_wait(req)

    def _send_goal_and_wait(self, goal: WorldGoal) -> WorldResult:
        self._client.send_goal_and_wait(goal)
        return self._client.get_result()

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
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.ADD
        req.body = make_world_body_box(size[0], size[1], size[2])
        req.parent_link_group = parent_link_group
        req.parent_link = parent_link
        req.pose = pose
        return self._send_goal_and_wait(req)

    def add_sphere(self,
                   name: str,
                   radius: float,
                   pose: PoseStamped,
                   parent_link: str = '',
                   parent_link_group: str = '') -> WorldResult:
        """
        See add_box.
        """
        world_body = WorldBody()
        world_body.type = WorldBody.PRIMITIVE_BODY
        world_body.shape.type = SolidPrimitive.SPHERE
        world_body.shape.dimensions.append(radius)
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.ADD
        req.body = world_body
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._send_goal_and_wait(req)

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
        world_body = WorldBody()
        world_body.type = WorldBody.MESH_BODY
        world_body.mesh = mesh
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.ADD
        req.body = world_body
        req.pose = pose
        req.body.scale.x = scale[0]
        req.body.scale.y = scale[1]
        req.body.scale.z = scale[2]
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._send_goal_and_wait(req)

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
        world_body = WorldBody()
        world_body.type = WorldBody.PRIMITIVE_BODY
        world_body.shape.type = SolidPrimitive.CYLINDER
        world_body.shape.dimensions = [0, 0]
        world_body.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT] = height
        world_body.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS] = radius
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.ADD
        req.body = world_body
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._send_goal_and_wait(req)

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
        :param timeout: how long to wait in case Giskard is busy processing a goal.
        :return: result message
        """
        req = WorldGoal()
        req.operation = WorldGoal.UPDATE_PARENT_LINK
        req.group_name = str(name)
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._send_goal_and_wait(req)

    def detach_group(self, object_name: str) -> WorldResult:
        """
        A wrapper for update_parent_link_of_group which set parent_link to the root link of the world.
        """
        req = WorldGoal()
        req.group_name = str(object_name)
        req.operation = req.UPDATE_PARENT_LINK
        return self._send_goal_and_wait(req)

    def add_urdf(self,
                 name: str,
                 urdf: str,
                 pose: PoseStamped,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 js_topic: Optional[str] = '') -> WorldResult:
        """
        Adds an urdf to the world.
        :param name: name the group containing the urdf will have.
        :param urdf: urdf as string, no path!
        :param pose: pose of the root link of the new object
        :param parent_link: to which link the urdf will be attached
        :param parent_link_group: if parent_link is not unique, search here for matches.
        :param js_topic: Giskard will listen on that topic for joint states and update the urdf accordingly
        :return: response message
        """
        js_topic = str(js_topic)
        urdf_body = WorldBody()
        urdf_body.type = WorldBody.URDF_BODY
        urdf_body.urdf = str(urdf)
        urdf_body.joint_state_topic = js_topic
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.ADD
        req.body = urdf_body
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._send_goal_and_wait(req)

    def get_control_mode(self) -> ControlModes:
        """
        returns the ControlMode of Giskard
        :return: ControlModes
        """
        rep: TriggerResponse = self._control_mode_srv.call(TriggerRequest())
        return ControlModes[rep.message]

    def dye_group(self, group_name: str, rgba: Tuple[float, float, float, float]) -> DyeGroupResponse:
        """
        Change the color of the ghost for this particular group.
        """
        req = DyeGroupRequest()
        req.group_name = group_name
        req.color.r = rgba[0]
        req.color.g = rgba[1]
        req.color.b = rgba[2]
        req.color.a = rgba[3]
        return self._dye_group_srv(req)

    def get_group_names(self) -> List[str]:
        """
        Returns the names of every group in the world.
        """
        resp: GetGroupNamesResponse = self._get_group_names_srv()
        return resp.group_names

    def get_group_info(self, group_name: str) -> GetGroupInfoResponse:
        """
        Returns the joint state, joint state topic and pose of a group.
        """
        req = GetGroupInfoRequest()
        req.group_name = group_name
        return self._get_group_info_srv.call(req)

    def get_controlled_joints(self, group_name: str) -> List[str]:
        """
        Returns all joints of a group that are flagged as controlled.
        """
        return self.get_group_info(group_name).controlled_joints

    def update_group_pose(self, group_name: str, new_pose: PoseStamped) -> WorldResult:
        """
        Overwrites the pose specified in the joint that connects the two groups.
        :param group_name: Name of the group that will move
        :param new_pose: New pose of the group
        :return: Giskard's reply
        """
        req = WorldGoal()
        req.operation = req.UPDATE_POSE
        req.group_name = group_name
        req.pose = new_pose
        res = self._send_goal_and_wait(req)
        if res.error.code == GiskardError.SUCCESS:
            return res
        if res.error.code == GiskardError.UNKNOWN_GROUP:
            raise UnknownGroupException(res.error.msg)
        raise ServiceException(res.error.msg)

    def register_group(self, new_group_name: str, root_link_name: str,
                       root_link_group_name: str) -> WorldResult:
        """
        Register a new group for reference in collision checking. All child links of root_link_name will belong to it.
        :param new_group_name: Name of the new group.
        :param root_link_name: root link of the new group
        :param root_link_group_name: Name of the group root_link_name belongs to
        :return: WorldResult
        """
        req = WorldGoal()
        req.operation = WorldGoal.REGISTER_GROUP
        req.group_name = new_group_name
        req.parent_link_group = root_link_group_name
        req.parent_link = root_link_name
        res = self._send_goal_and_wait(req)
        if res.error.code == GiskardError.DUPLICATE_NAME:
            raise DuplicateNameException(f'Group with name {new_group_name} already exists.')
        return res


class MotionGoalWrapper:
    _goals: List[MotionGoal]
    _collision_entries: Dict[Tuple[str, str, str], List[CollisionEntry]]
    avoid_name_conflict: bool

    def __init__(self, robot_name: str, avoid_name_conflict: bool = False):
        self.robot_name = robot_name
        self.reset()
        self.avoid_name_conflict = avoid_name_conflict

    def reset(self):
        """
        Clears all goals.
        """
        self._goals = []
        self._collision_entries = defaultdict(list)

    def get_goals(self) -> List[MotionGoal]:
        self._add_collision_entries_as_goals()
        return self._goals

    def number_of_goals(self) -> int:
        return len(self._goals)

    def add_motion_goal(self, *,
                        motion_goal_class: str,
                        name: Optional[str] = None,
                        start_condition: str = '',
                        hold_condition: str = '',
                        end_condition: str = '',
                        **kwargs):
        """
        Generic function to add a motion goal.
        :param motion_goal_class: Name of a class defined in src/giskardpy/goals
        :param name: a unique name for the goal, will use class name by default
        :param start_condition: a logical expression to define the start condition for this monitor. e.g.
                                    not 'monitor1' and ('monitor2' or 'monitor3')
        :param hold_condition: a logical expression. Goal will be on hold if it is True and active otherwise
        :param end_condition: a logical expression. Goal will become inactive when this becomes True.
        :param kwargs: kwargs for __init__ function of motion_goal_class
        """
        name = name or motion_goal_class
        if self.avoid_name_conflict:
            name = f'G{self.number_of_goals()} {name}'
        motion_goal = MotionGoal()
        motion_goal.name = name
        motion_goal.motion_goal_class = motion_goal_class
        motion_goal.start_condition = start_condition
        motion_goal.hold_condition = hold_condition
        motion_goal.end_condition = end_condition
        motion_goal.kwargs = kwargs_to_json(kwargs)
        self._goals.append(motion_goal)

    def _add_collision_avoidance(self,
                                 collisions: List[CollisionEntry],
                                 start_condition: str = '',
                                 hold_condition: str = '',
                                 end_condition: str = ''):
        key = (start_condition, hold_condition, end_condition)
        self._collision_entries[key].extend(collisions)

    def _add_collision_entries_as_goals(self):
        for (start_condition, hold_condition, end_condition), collision_entries in self._collision_entries.items():
            name = 'collision avoidance'
            if start_condition or hold_condition or end_condition:
                name += f'{start_condition}, {hold_condition}, {end_condition}'
            self.add_motion_goal(motion_goal_class=CollisionAvoidance.__name__,
                                 name=name,
                                 collision_entries=collision_entries,
                                 start_condition=start_condition,
                                 hold_condition=hold_condition,
                                 end_condition=end_condition)

    def allow_collision(self,
                        group1: str = CollisionEntry.ALL,
                        group2: str = CollisionEntry.ALL,
                        start_condition: str = '',
                        hold_condition: str = '',
                        end_condition: str = ''):
        """
        Tell Giskard to allow collision between group1 and group2. Use CollisionEntry. ALL to allow collision with all
        groups.
        :param group1: name of the first group
        :param group2: name of the second group
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = str(group1)
        collision_entry.group2 = str(group2)
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      hold_condition=hold_condition,
                                      end_condition=end_condition)

    def avoid_collision(self,
                        min_distance: Optional[float] = None,
                        group1: str = CollisionEntry.ALL,
                        group2: str = CollisionEntry.ALL,
                        start_condition: str = '',
                        hold_condition: str = '',
                        end_condition: str = ''):
        """
        Tell Giskard to avoid collision between group1 and group2. Use CollisionEntry. ALL to allow collision with all
        groups.
        :param min_distance: set this to overwrite the default distances
        :param group1: name of the first group
        :param group2: name of the second group
        """
        if min_distance is None:
            min_distance = - 1
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        collision_entry.group1 = group1
        collision_entry.group2 = group2
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      hold_condition=hold_condition,
                                      end_condition=end_condition)

    def allow_all_collisions(self,
                             start_condition: str = '',
                             hold_condition: str = '',
                             end_condition: str = ''):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      hold_condition=hold_condition,
                                      end_condition=end_condition)

    def avoid_all_collisions(self,
                             min_distance: Optional[float] = None,
                             start_condition: str = '',
                             hold_condition: str = '',
                             end_condition: str = ''):
        """
        If you don't want to override the distance, don't call this function. Avoid all is the default, if you don't
        add any collision entries.
        :param min_distance: set this to overwrite default distances
        """
        if min_distance is None:
            min_distance = -1
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      hold_condition=hold_condition,
                                      end_condition=end_condition)

    def allow_self_collision(self,
                             robot_name: Optional[str] = None,
                             start_condition: str = '',
                             hold_condition: str = '',
                             end_condition: str = ''):
        """
        Allows the collision of the robot with itself for the next goal.
        :param robot_name: if there are multiple robots, specify which one.
        """
        if robot_name is None:
            robot_name = self.robot_name
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = robot_name
        collision_entry.group2 = robot_name
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      hold_condition=hold_condition,
                                      end_condition=end_condition)

    def add_joint_position(self,
                           goal_state: Dict[str, float],
                           group_name: Optional[str] = None,
                           weight: Optional[float] = None,
                           max_velocity: Optional[float] = None,
                           name: Optional[str] = None,
                           start_condition: str = '',
                           hold_condition: str = '',
                           end_condition: str = '',
                           **kwargs: goal_parameter):
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight: None = use default weight
        :param max_velocity: will be applied to all joints
        """
        self.add_motion_goal(motion_goal_class=JointPositionList.__name__,
                             goal_state=goal_state,
                             group_name=group_name,
                             weight=weight,
                             max_velocity=max_velocity,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_cartesian_pose(self,
                           goal_pose: PoseStamped,
                           tip_link: str,
                           root_link: str,
                           tip_group: Optional[str] = None,
                           root_group: Optional[str] = None,
                           reference_linear_velocity: Optional[float] = None,
                           reference_angular_velocity: Optional[float] = None,
                           absolute: bool = False,
                           weight: Optional[float] = None,
                           name: Optional[str] = None,
                           start_condition: str = '',
                           hold_condition: str = '',
                           end_condition: str = '',
                           **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link to the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: None = use default weight
        """
        self.add_motion_goal(motion_goal_class=CartesianPose.__name__,
                             goal_pose=goal_pose,
                             tip_link=tip_link,
                             root_link=root_link,
                             root_group=root_group,
                             tip_group=tip_group,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             weight=weight,
                             name=name,
                             absolute=absolute,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_align_planes(self,
                         goal_normal: Vector3Stamped,
                         tip_link: str,
                         tip_normal: Vector3Stamped,
                         root_link: str,
                         tip_group: str = None,
                         root_group: str = None,
                         reference_angular_velocity: Optional[float] = None,
                         weight: Optional[float] = None,
                         name: Optional[str] = None,
                         start_condition: str = '',
                         hold_condition: str = '',
                         end_condition: str = '',
                         **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between tip and root to align tip_normal with goal_normal.
        :param goal_normal:
        :param tip_link: tip link of the kinematic chain
        :param tip_normal:
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip_link is not unique, search in this group for matches.
        :param root_group: if root_link is not unique, search in this group for matches.
        :param reference_angular_velocity: rad/s
        :param weight:
        """
        self.add_motion_goal(motion_goal_class=AlignPlanes.__name__,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             tip_normal=tip_normal,
                             root_link=root_link,
                             root_group=root_group,
                             goal_normal=goal_normal,
                             max_angular_velocity=reference_angular_velocity,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_avoid_joint_limits(self,
                               percentage: int = 15,
                               joint_list: Optional[List[str]] = None,
                               weight: Optional[float] = None,
                               name: Optional[str] = None,
                               start_condition: str = '',
                               hold_condition: str = '',
                               end_condition: str = ''):
        """
        This goal will push joints away from their position limits. For example if percentage is 15 and the joint
        limits are 0-100, it will push it into the 15-85 range.
        """
        self.add_motion_goal(motion_goal_class=AvoidJointLimits.__name__,
                             percentage=percentage,
                             weight=weight,
                             joint_list=joint_list,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_close_container(self,
                            tip_link: str,
                            environment_link: str,
                            tip_group: Optional[str] = None,
                            environment_group: Optional[str] = None,
                            goal_joint_state: Optional[float] = None,
                            weight: Optional[float] = None,
                            name: Optional[str] = None,
                            start_condition: str = '',
                            hold_condition: str = '',
                            end_condition: str = ''):
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        self.add_motion_goal(motion_goal_class=Close.__name__,
                             tip_link=tip_link,
                             environment_link=environment_link,
                             tip_group=tip_group,
                             environment_group=environment_group,
                             goal_joint_state=goal_joint_state,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_open_container(self,
                           tip_link: str,
                           environment_link: str,
                           special_door: Optional[bool] = False,
                           tip_group: Optional[str] = None,
                           environment_group: Optional[str] = None,
                           goal_joint_state: Optional[float] = None,
                           weight: Optional[float] = None,
                           name: Optional[str] = None,
                           start_condition: str = '',
                           hold_condition: str = '',
                           end_condition: str = ''):
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
        self.add_motion_goal(motion_goal_class=Open.__name__,
                             tip_link=tip_link,
                             environment_link=environment_link,
                             tip_group=tip_group,
                             special_door=special_door,
                             environment_group=environment_group,
                             goal_joint_state=goal_joint_state,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_align_to_push_door(self,
                               root_link: str,
                               tip_link: str,
                               door_object: str,
                               door_handle: str,
                               tip_gripper_axis: Vector3Stamped,
                               weight: float,
                               tip_group: Optional[str] = None,
                               root_group: Optional[str] = None,
                               intermediate_point_scale: Optional[float] = 1,
                               name: Optional[str] = None,
                               start_condition: str = '',
                               hold_condition: str = '',
                               end_condition: str = ''):
        """
        Aligns the tip_link with the door_object to push it open. Only works if the door object is part of the urdf.
        The door has to be open a little before aligning.
        : param root_link: root link of the kinematic chain
        : param tip_link: end effector
        : param door object: name of the object to be pushed
        : param door_height: height of the door
        : param door_handle: name of the object handle
        : param object_joint_name: name of the joint that rotates
        : param tip_gripper_axis: axis of the tip_link that will be aligned along the door rotation axis
        : param object_rotation_axis: door rotation axis w.r.t root
        """
        self.add_motion_goal(motion_goal_class=AlignToPushDoor.__name__,
                             root_link=root_link,
                             tip_link=tip_link,
                             door_handle=door_handle,
                             door_object=door_object,
                             tip_gripper_axis=tip_gripper_axis,
                             tip_group=tip_group,
                             root_group=root_group,
                             intermediate_point_scale=intermediate_point_scale,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_pre_push_door(self,
                          root_link: str,
                          tip_link: str,
                          door_object: str,
                          door_handle: str,
                          weight: float,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          reference_linear_velocity: Optional[float] = None,
                          reference_angular_velocity: Optional[float] = None,
                          name: Optional[str] = None,
                          start_condition: str = '',
                          hold_condition: str = '',
                          end_condition: str = ''):
        """
        Positions the gripper in contact with the door before pushing to open.
        : param root_link: root link of the kinematic chain
        : param tip_link: end effector
        : param door object: name of the object to be pushed
        : param door_handle: name of the object handle
        : param root_V_object_rotation_axis: door rotation axis w.r.t root
        : param root_V_object_normal: door normal w.r.t root
        """
        self.add_motion_goal(motion_goal_class=PrePushDoor.__name__,
                             root_link=root_link,
                             tip_link=tip_link,
                             door_object=door_object,
                             door_handle=door_handle,
                             tip_group=tip_group,
                             root_group=root_group,
                             weight=weight,
                             name=name,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_diff_drive_base(self,
                            goal_pose: PoseStamped,
                            tip_link: str,
                            root_link: str,
                            tip_group: Optional[str] = None,
                            root_group: Optional[str] = None,
                            reference_linear_velocity: Optional[float] = None,
                            reference_angular_velocity: Optional[float] = None,
                            weight: Optional[float] = None,
                            name: Optional[str] = None,
                            start_condition: str = '',
                            hold_condition: str = '',
                            end_condition: str = '',
                            **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        It is specifically for differential drives. Will drive towards the goal the following way:
        1. orient to goal
        2. drive to goal position in a straight line
        3. orient to goal orientation
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        """
        self.add_motion_goal(motion_goal_class=DiffDriveBaseGoal.__name__,
                             goal_pose=goal_pose,
                             tip_link=tip_link,
                             root_link=root_link,
                             root_group=root_group,
                             tip_group=tip_group,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_grasp_bar(self,
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
                      weight: Optional[float] = None,
                      name: Optional[str] = None,
                      start_condition: str = '',
                      hold_condition: str = '',
                      end_condition: str = '',
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
        """
        self.add_motion_goal(motion_goal_class=GraspBar.__name__,
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
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_limit_cartesian_velocity(self,
                                     tip_link: str,
                                     root_link: str,
                                     tip_group: Optional[str] = None,
                                     root_group: Optional[str] = None,
                                     max_linear_velocity: float = 0.1,
                                     max_angular_velocity: float = 0.5,
                                     weight: Optional[float] = None,
                                     hard: bool = False,
                                     name: Optional[str] = None,
                                     start_condition: str = '',
                                     hold_condition: str = '',
                                     end_condition: str = '',
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
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        self.add_motion_goal(motion_goal_class=CartesianVelocityLimit.__name__,
                             root_link=root_link,
                             root_group=root_group,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             weight=weight,
                             max_linear_velocity=max_linear_velocity,
                             max_angular_velocity=max_angular_velocity,
                             hard=hard,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_pointing(self,
                     goal_point: PointStamped,
                     tip_link: str,
                     pointing_axis: Vector3Stamped,
                     root_link: str,
                     tip_group: Optional[str] = None,
                     root_group: Optional[str] = None,
                     max_velocity: float = 0.3,
                     weight: Optional[float] = None,
                     name: Optional[str] = None,
                     start_condition: str = '',
                     hold_condition: str = '',
                     end_condition: str = '',
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
        """
        self.add_motion_goal(motion_goal_class=Pointing.__name__,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             goal_point=goal_point,
                             root_link=root_link,
                             root_group=root_group,
                             pointing_axis=pointing_axis,
                             max_velocity=max_velocity,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_real_time_pointing(self,
                               tip_link: str,
                               pointing_axis: Vector3Stamped,
                               root_link: str,
                               topic_name: str,
                               tip_group: Optional[str] = None,
                               root_group: Optional[str] = None,
                               max_velocity: float = 0.3,
                               weight: Optional[float] = None,
                               name: Optional[str] = None,
                               start_condition: str = '',
                               hold_condition: str = '',
                               end_condition: str = '',
                               **kwargs: goal_parameter):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param topic_name: name of a topic of type PointStamped
        :param root_link: root link of the kinematic chain.
        :param tip_group: if tip_link is not unique, search this group for matches.
        :param root_group: if root_link is not unique, search this group for matches.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        """
        self.add_motion_goal(motion_goal_class=RealTimePointing.__name__,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             root_link=root_link,
                             topic_name=topic_name,
                             root_group=root_group,
                             pointing_axis=pointing_axis,
                             max_velocity=max_velocity,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_carry_my_luggage(self,
                             name: str,
                             tracked_human_position_topic_name: str = '/robokudovanessa/human_position',
                             laser_topic_name: str = '/hsrb/base_scan',
                             point_cloud_laser_topic_name: Optional[str] = None,
                             odom_joint_name: str = 'brumbrum',
                             root_link: Optional[str] = None,
                             camera_link: str = 'head_rgbd_sensor_link',
                             distance_to_target_stop_threshold: float = 1,
                             laser_scan_age_threshold: float = 2,
                             laser_distance_threshold: float = 0.5,
                             laser_distance_threshold_width: float = 0.8,
                             laser_avoidance_angle_cutout: float = np.pi / 4,
                             laser_avoidance_sideways_buffer: float = 0.04,
                             base_orientation_threshold: float = np.pi / 16,
                             tracked_human_position_topic_name_timeout: int = 30,
                             max_rotation_velocity: float = 0.5,
                             max_rotation_velocity_head: float = 1,
                             max_translation_velocity: float = 0.38,
                             traj_tracking_radius: float = 0.4,
                             height_for_camera_target: float = 1,
                             laser_frame_id: str = 'base_range_sensor_link',
                             target_age_threshold: float = 2,
                             target_age_exception_threshold: float = 5,
                             clear_path: bool = False,
                             drive_back: bool = False,
                             enable_laser_avoidance: bool = True,
                             start_condition: str = '',
                             hold_condition: str = '',
                             end_condition: str = ''):
        """
        :param name: name of the goal
        :param tracked_human_position_topic_name: name of the topic where the tracked human is published
        :param laser_topic_name: topic name of the laser scanner
        :param point_cloud_laser_topic_name: topic name of a second laser scanner, e.g. from a point cloud to laser scanner node
        :param odom_joint_name: name of the odom joint
        :param root_link: will use global reference frame
        :param camera_link: link of the camera that will point to the tracked human
        :param distance_to_target_stop_threshold: will pause if closer than this many meter to the target
        :param laser_scan_age_threshold: giskard will complain if scans are older than this many seconds
        :param laser_distance_threshold: this and width are used to crate a stopping zone around the robot.
                                            laser distance draws a circle around the robot and width lines to the left and right.
                                            the stopping zone is the minimum of the two.
        :param laser_distance_threshold_width: see laser_distance_threshold
        :param laser_avoidance_angle_cutout: if something is in the stop zone in front of the robot in +/- this angle range
                                                giskard will pause, otherwise it will try to dodge left or right
        :param laser_avoidance_sideways_buffer: increase this if the robot is shaking too much if something is to its
                                                left and right at the same time.
        :param base_orientation_threshold: giskard will align the base of the robot to the target, this is a +/- buffer to avoid shaking
        :param tracked_human_position_topic_name_timeout: on start up, wait this long for tracking msg to arrive
        :param max_rotation_velocity: how quickly the base can change orientation
        :param max_rotation_velocity_head: how quickly the head rotates
        :param max_translation_velocity: how quickly the base drives
        :param traj_tracking_radius: how close the robots root link will try to stick to the path in meter
        :param height_for_camera_target: target tracking with head will ignore the published height, but use this instead
        :param laser_frame_id: frame_id of the laser scanner
        :param target_age_threshold: will stop looking at the target if the messages are older than this many seconds
        :param target_age_exception_threshold: if there are no messages from the tracked_human_position_topic_name
                                                            topic for this many seconds, cancel
        :param clear_path: clear the saved path. if called repeated will, giskard would just continue the old path if not cleared
        :param drive_back: follow the saved path to drive back
        :param enable_laser_avoidance:
        :param start_condition:
        :param hold_condition:
        :param end_condition:
        """
        self.add_motion_goal(motion_goal_class=CarryMyBullshit.__name__,
                             name=name,
                             patrick_topic_name=tracked_human_position_topic_name,
                             laser_topic_name=laser_topic_name,
                             point_cloud_laser_topic_name=point_cloud_laser_topic_name,
                             odom_joint_name=odom_joint_name,
                             root_link=root_link,
                             camera_link=camera_link,
                             distance_to_target_stop_threshold=distance_to_target_stop_threshold,
                             laser_scan_age_threshold=laser_scan_age_threshold,
                             laser_distance_threshold=laser_distance_threshold,
                             laser_distance_threshold_width=laser_distance_threshold_width,
                             laser_avoidance_angle_cutout=laser_avoidance_angle_cutout,
                             laser_avoidance_sideways_buffer=laser_avoidance_sideways_buffer,
                             base_orientation_threshold=base_orientation_threshold,
                             wait_for_patrick_timeout=tracked_human_position_topic_name_timeout,
                             max_rotation_velocity=max_rotation_velocity,
                             max_rotation_velocity_head=max_rotation_velocity_head,
                             max_translation_velocity=max_translation_velocity,
                             traj_tracking_radius=traj_tracking_radius,
                             height_for_camera_target=height_for_camera_target,
                             laser_frame_id=laser_frame_id,
                             target_age_threshold=target_age_threshold,
                             target_age_exception_threshold=target_age_exception_threshold,
                             clear_path=clear_path,
                             drive_back=drive_back,
                             enable_laser_avoidance=enable_laser_avoidance,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_follow_nav_path(self,
                            name: str,
                            path: Path,
                            laser_topics: Tuple[str] = ('/hsrb/base_scan',),
                            odom_joint_name: Optional[str] = None,
                            root_link: Optional[str] = None,
                            camera_link: str = 'head_rgbd_sensor_link',
                            distance_to_target_stop_threshold: float = 1,
                            laser_scan_age_threshold: float = 2,
                            laser_distance_threshold: float = 0.5,
                            laser_distance_threshold_width: float = 0.8,
                            laser_avoidance_angle_cutout: float = np.pi / 4,
                            laser_avoidance_sideways_buffer: float = 0.04,
                            base_orientation_threshold: float = np.pi / 16,
                            max_rotation_velocity: float = 0.5,
                            max_rotation_velocity_head: float = 1,
                            max_translation_velocity: float = 0.38,
                            traj_tracking_radius: float = 0.4,
                            height_for_camera_target: float = 1,
                            laser_frame_id: str = 'base_range_sensor_link',
                            start_condition: str = '',
                            hold_condition: str = '',
                            end_condition: str = ''):
        """
        Will follow the path, orienting itself and the head towards the next points in the list.
        At the end orient itself according to the final orientation in it. All other orientations will be ignored.
        :param name: name of the goal
        :param path: a nav path, make sure it's ordered correctly!
        :param odom_joint_name: name of the odom joint
        :param root_link: will use global reference frame
        :param camera_link: link of the camera that will point to the tracked human
        :param laser_scan_age_threshold: giskard will complain if scans are older than this many seconds
        :param laser_distance_threshold: this and width are used to crate a stopping zone around the robot.
                                            laser distance draws a circle around the robot and width lines to the left and right.
                                            the stopping zone is the minimum of the two.
        :param laser_distance_threshold_width: see laser_distance_threshold
        :param laser_avoidance_angle_cutout: if something is in the stop zone in front of the robot in +/- this angle range
                                                giskard will pause, otherwise it will try to dodge left or right
        :param laser_avoidance_sideways_buffer: increase this if the robot is shaking too much if something is to its
                                                left and right at the same time.
        :param base_orientation_threshold: giskard will align the base of the robot to the target, this is a +/- buffer to avoid shaking
        :param max_rotation_velocity: how quickly the base can change orientation
        :param max_rotation_velocity_head: how quickly the head rotates
        :param max_translation_velocity: how quickly the base drives
        :param traj_tracking_radius: how close the robots root link will try to stick to the path in meter
        :param height_for_camera_target: target tracking with head will ignore the published height, but use this instead
        :param laser_frame_id: frame_id of the laser scanner
        :param start_condition:
        :param hold_condition:
        :param end_condition:
        """
        self.add_motion_goal(motion_goal_class=FollowNavPath.__name__,
                             name=name,
                             path=path,
                             laser_topics=laser_topics,
                             odom_joint_name=odom_joint_name,
                             root_link=root_link,
                             camera_link=camera_link,
                             laser_scan_age_threshold=laser_scan_age_threshold,
                             laser_distance_threshold=laser_distance_threshold,
                             laser_distance_threshold_width=laser_distance_threshold_width,
                             laser_avoidance_angle_cutout=laser_avoidance_angle_cutout,
                             laser_avoidance_sideways_buffer=laser_avoidance_sideways_buffer,
                             base_orientation_threshold=base_orientation_threshold,
                             max_rotation_velocity=max_rotation_velocity,
                             max_rotation_velocity_head=max_rotation_velocity_head,
                             max_translation_velocity=max_translation_velocity,
                             traj_tracking_radius=traj_tracking_radius,
                             height_for_camera_target=height_for_camera_target,
                             laser_frame_id=laser_frame_id,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_cartesian_orientation(self,
                                  goal_orientation: QuaternionStamped,
                                  tip_link: str,
                                  root_link: str,
                                  tip_group: Optional[str] = None,
                                  root_group: Optional[str] = None,
                                  reference_velocity: Optional[float] = None,
                                  weight: Optional[float] = None,
                                  absolute: bool = False,
                                  name: Optional[str] = None,
                                  start_condition: str = '',
                                  hold_condition: str = '',
                                  end_condition: str = '',
                                  **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_orientation.
        :param goal_orientation:
        :param tip_link: tip link of kinematic chain
        :param root_link: root link of kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param reference_velocity: rad/s, approx limit
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        """
        self.add_motion_goal(motion_goal_class=CartesianOrientation.__name__,
                             goal_orientation=goal_orientation,
                             tip_link=tip_link,
                             root_link=root_link,
                             tip_group=tip_group,
                             root_group=root_group,
                             reference_velocity=reference_velocity,
                             weight=weight,
                             absolute=absolute,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def set_seed_configuration(self,
                               seed_configuration: Dict[str, float],
                               group_name: Optional[str] = None,
                               name: Optional[str] = None):
        """
        Only meant for use with projection. Changes the world state to seed_configuration before starting planning,
        without having to plan a motion to it like with add_joint_position
        """
        raise DeprecationWarning('please use monitors.set_seed_configuration instead')

    def set_seed_odometry(self,
                          base_pose: PoseStamped,
                          group_name: Optional[str] = None,
                          name: Optional[str] = None):
        """
        Only meant for use with projection. Overwrites the odometry transform with base_pose.
        """
        raise DeprecationWarning('please use monitors.set_seed_odometry instead')

    def add_cartesian_pose_straight(self,
                                    goal_pose: PoseStamped,
                                    tip_link: str,
                                    root_link: str,
                                    tip_group: Optional[str] = None,
                                    root_group: Optional[str] = None,
                                    reference_linear_velocity: Optional[float] = None,
                                    reference_angular_velocity: Optional[float] = None,
                                    weight: Optional[float] = None,
                                    absolute: bool = False,
                                    name: Optional[str] = None,
                                    start_condition: str = '',
                                    hold_condition: str = '',
                                    end_condition: str = '',
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
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        """
        self.add_motion_goal(motion_goal_class=CartesianPoseStraight.__name__,
                             goal_pose=goal_pose,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             root_link=root_link,
                             root_group=root_group,
                             weight=weight,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             name=name,
                             absolute=absolute,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_cartesian_position(self,
                               goal_point: PointStamped,
                               tip_link: str,
                               root_link: str,
                               tip_group: Optional[str] = None,
                               root_group: Optional[str] = None,
                               reference_velocity: Optional[float] = 0.2,
                               weight: Optional[float] = None,
                               absolute: bool = False,
                               name: Optional[str] = None,
                               start_condition: str = '',
                               hold_condition: str = '',
                               end_condition: str = '',
                               **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_point.
        :param goal_point: goal point
        :param tip_link: tip link of the kinematic chain
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param reference_velocity: m/s
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        """
        self.add_motion_goal(motion_goal_class=CartesianPosition.__name__,
                             goal_point=goal_point,
                             tip_link=tip_link,
                             root_link=root_link,
                             tip_group=tip_group,
                             root_group=root_group,
                             reference_velocity=reference_velocity,
                             weight=weight,
                             absolute=absolute,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_cartesian_position_straight(self,
                                        goal_point: PointStamped,
                                        tip_link: str,
                                        root_link: str,
                                        tip_group: Optional[str] = None,
                                        root_group: Optional[str] = None,
                                        reference_velocity: float = None,
                                        weight: Optional[float] = None,
                                        absolute: bool = False,
                                        name: Optional[str] = None,
                                        start_condition: str = '',
                                        hold_condition: str = '',
                                        end_condition: str = '',
                                        **kwargs: goal_parameter):
        """
        Same as set_translation_goal, but will try to move in a straight line.
        """
        self.add_motion_goal(motion_goal_class=CartesianPositionStraight.__name__,
                             goal_point=goal_point,
                             tip_link=tip_link,
                             root_link=root_link,
                             tip_group=tip_group,
                             root_group=root_group,
                             reference_velocity=reference_velocity,
                             weight=weight,
                             name=name,
                             absolute=absolute,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_grasp_bar_offset(self,
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
                             weight: Optional[float] = None,
                             name: Optional[str] = None,
                             start_condition: str = '',
                             hold_condition: str = '',
                             end_condition: str = '',
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
        :param grasp_axis_offset: offset of the grasp axis
        :param bar_length: length of the bar to be grasped
        :param root_group: if root_link is not unique, search in this group for matches
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight:
        :param name:
        """
        self.add_motion_goal(motion_goal_class=GraspBarOffset.__name__,
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
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def hsrb_open_door_goal(self,
                            door_handle_link: str,
                            tip_link: str = 'hand_gripper_tool_frame',
                            name: str = 'HSRB_open_door',
                            handle_limit: Optional[float] = (np.pi / 6),
                            hinge_limit: Optional[float] = -(np.pi / 4)):
        """
        HSRB specific open door goal wrapper

        :param door_handle_link: Link of the door handle
        :param tip_link: Link that's grasping the door handle
        :param name: name of the goal for distinction between same goals
        :param handle_limit: Limits the handle opening to given value
        :param hinge_limit: Limits the hinge opening to given value
        """

        self.add_open_door_goal(tip_link=tip_link,
                                door_handle_link=door_handle_link,
                                name=name,
                                handle_limit=handle_limit,
                                hinge_limit=hinge_limit)

    def hsrb_door_handle_grasp(self,
                               handle_name: str,
                               handle_bar_length: float = 0,
                               tip_link: str = 'hand_gripper_tool_frame',
                               root_link: str = 'map',
                               bar_axis_v: Optional[Vector3] = None,
                               tip_grasp_axis_v: Optional[Vector3] = None):
        """
        HSRB specific set_grasp_bar_goal, that only needs handle_name of the door_handle

        :param handle_name: URDF link that represents the door handle
        :param handle_bar_length: length of the door handle
        :param tip_link: robot link, that grasps the handle
        :param root_link: root link of the kinematic chain
        :param bar_axis_v: Vector for changing the orientation of the door handle
        :param tip_grasp_axis_v: Vector for the orientation of the tip grasp link
        """
        # TODO: actually use Vector3Stamped as parameter instead of Vector3, make handle_name and bar_axis/tip_grasp_axis mutally exclusive
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_name
        if bar_axis_v is None:
            bar_axis.vector = Vector3(0, 1, 0)
        else:
            bar_axis.vector = bar_axis_v

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = tip_link
        if tip_grasp_axis_v is None:
            tip_grasp_axis.vector = Vector3(1, 0, 0)
        else:
            tip_grasp_axis.vector = tip_grasp_axis_v

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_name
        bar_center.point.y = 0.045

        self.add_grasp_bar(root_link=root_link,
                           tip_link=tip_link,
                           tip_grasp_axis=tip_grasp_axis,
                           bar_center=bar_center,
                           bar_axis=bar_axis,
                           bar_length=handle_bar_length)

    def hsrb_dishwasher_door_around(self,
                                    handle_name: str,
                                    root_link: str = 'map',
                                    tip_link: str = 'hand_gripper_tool_frame'):
        """
        HSRB specific avoid dishwasher door goal

        :param handle_name: name of the dishwasher handle
        :param tip_link: robot link, that grasps the handle
        :param root_link: root link of the kinematic chain
        """

        self.add_motion_goal(motion_goal_class=MoveAroundDishwasher.__name__,
                             handle_name=handle_name,
                             root_link=root_link,
                             tip_link=tip_link)

    def hsrb_align_to_push_door_goal(self,
                                     handle_name: str,
                                     hinge_frame_id: str,
                                     tip_link: str = 'hand_gripper_tool_frame',
                                     root_link: str = 'map',
                                     weight: float = WEIGHT_ABOVE_CA):
        """
        HSRB specific push door open goal of dishwasher

        :param handle_name: name of the door handle
        :param hinge_frame_id: Frame id of the door hinge
        :param tip_link: robot link, that grasps the handle
        :param root_link: root link of the kinematic chain
        :param weight: Weight of the goal compared to Collision Avoidance
        """

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = tip_link
        tip_grasp_axis.vector.y = 1

        self.add_align_to_push_door(root_link=root_link,
                                    tip_link=tip_link,
                                    door_handle=handle_name,
                                    door_object=hinge_frame_id,
                                    tip_gripper_axis=tip_grasp_axis,
                                    weight=weight,
                                    intermediate_point_scale=0.95)

    def hsrb_pre_push_door_goal(self,
                                handle_name: str,
                                hinge_frame_id: str,
                                root_link: str = 'map',
                                tip_link: str = 'hand_gripper_tool_frame',
                                weight: float = WEIGHT_ABOVE_CA):
        """
        HSRB specific pre push door open goal of dishwasher

        :param handle_name: name of the door handle
        :param hinge_frame_id: Frame id of the door hinge
        :param tip_link: robot link, that grasps the handle
        :param root_link: root link of the kinematic chain
        :param weight: Weight of the goal compared to Collision Avoidance
        """

        self.add_pre_push_door(root_link=root_link,
                               tip_link=tip_link,
                               door_handle=handle_name,
                               weight=weight,
                               door_object=hinge_frame_id)

    def open_container_goal(self,
                            tip_link: str,
                            environment_link: str,
                            special_door: Optional[bool] = False,
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
        :param special_door: workaround for dishwasher opening motion
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param environment_group: if environment_link is not unique, search in this group for matches
        :param goal_joint_state: goal state for the container. default is maximum joint state.
        :param weight:
        """
        self.add_open_container(tip_link=tip_link,
                                environment_link=environment_link,
                                tip_group=tip_group,
                                special_door=special_door,
                                environment_group=environment_group,
                                goal_joint_state=goal_joint_state,
                                start_condition=start_condition,
                                hold_condition=hold_condition,
                                end_condition=end_condition,
                                weight=weight)

    # TODO: change object_size Vector3Stamped instead
    def add_reaching(self,
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

        self.add_motion_goal(motion_goal_class=Reaching.__name__,
                             grasp=grasp,
                             align=align,
                             object_name=object_name,
                             object_shape=object_shape,
                             goal_pose=goal_pose,
                             object_size=object_size,
                             root_link=root_link,
                             tip_link=tip_link,
                             velocity=velocity)

    def add_placing(self,
                    context,
                    goal_pose: PoseStamped,
                    tip_link: str = 'hand_palm_link',
                    velocity: float = 0.02):

        self.add_motion_goal(motion_goal_class=Placing.__name__,
                             context=context,
                             goal_pose=goal_pose,
                             tip_link=tip_link,
                             velocity=velocity)

    def add_vertical_motion(self,
                            action: str,
                            distance: float = 0.02,
                            root_link: str = 'base_link',
                            tip_link: str = 'hand_palm_link'):

        self.add_motion_goal(motion_goal_class=VerticalMotion.__name__,
                             action=action,
                             distance=distance,
                             root_link=root_link,
                             tip_link=tip_link)

    def add_retract(self,
                    object_name: str,
                    distance: float = 0.1,
                    reference_frame: str = 'base_link',
                    root_link: str = 'map',
                    tip_link: str = 'base_link',
                    velocity: float = 0.2):

        self.add_motion_goal(motion_goal_class=Retracting.__name__,
                             object_name=object_name,
                             distance=distance,
                             reference_frame=reference_frame,
                             root_link=root_link,
                             tip_link=tip_link,
                             velocity=velocity)

    def add_align_height(self,
                         object_name: str,
                         goal_pose: PoseStamped,
                         object_height: float,
                         from_above: bool = False,
                         root_link: str = 'map',
                         tip_link: str = 'hand_gripper_tool_frame'):

        self.add_motion_goal(motion_goal_class=AlignHeight.__name__,
                             from_above=from_above,
                             object_name=object_name,
                             goal_pose=goal_pose,
                             object_height=object_height,
                             root_link=root_link,
                             tip_link=tip_link)

    def add_test_goal(self,
                      goal_name: str,
                      **kwargs):

        self.add_motion_goal(motion_goal_class=goal_name,
                             **kwargs)

    def add_take_pose(self,
                      pose_keyword: str):

        self.add_motion_goal(motion_goal_class=TakePose.__name__,
                             pose_keyword=pose_keyword)

    def add_tilting(self,
                    tilt_direction: Optional[str] = None,
                    tilt_angle: Optional[float] = None,
                    tip_link: str = 'wrist_roll_joint',
                    ):

        self.add_motion_goal(motion_goal_class=Tilting.__name__,
                             direction=tilt_direction,
                             angle=tilt_angle,
                             tip_link=tip_link)

    def add_joint_rotation_continuous(self,
                                      joint_name: str,
                                      joint_center: float,
                                      joint_range: float,
                                      trajectory_length: float = 20,
                                      target_speed: float = 1,
                                      period_length: float = 1.0):

        self.add_motion_goal(motion_goal_class=JointRotationGoalContinuous.__name__,
                             joint_name=joint_name,
                             joint_center=joint_center,
                             joint_range=joint_range,
                             trajectory_length=trajectory_length,
                             target_speed=target_speed,
                             period_length=period_length)

    def add_mixing(self,
                   mixing_time=20,
                   weight: float = WEIGHT_ABOVE_CA):

        self.add_motion_goal(motion_goal_class=Mixing.__name__,
                             mixing_time=mixing_time,
                             weight=weight)

    def add_open_environment(self,
                             tip_link: str,
                             environment_link: str,
                             tip_group: Optional[str] = None,
                             environment_group: Optional[str] = None,
                             goal_joint_state: Optional[float] = None,
                             weight: float = WEIGHT_ABOVE_CA):

        self.add_motion_goal(motion_goal_class=Open.__name__,
                             tip_link=tip_link,
                             environment_link=environment_link,
                             tip_group=tip_group,
                             environment_group=environment_group,
                             goal_joint_state=goal_joint_state,
                             weight=weight)

    def add_open_door_goal(self,
                           tip_link: str,
                           door_handle_link: str,
                           name: str = None,
                           handle_limit: Optional[float] = None,
                           hinge_limit: Optional[float] = None):
        """
        Adds OpenDoorGoal to motion goal execution plan

        :param tip_link: Link that is grasping the door handle
        :param door_handle_link: Link of the door handle of the door that is to be opened
        :param name: Name of the Goal for distinction between similar goals
        :param handle_limit: Limits the handle opening to given value
        :param hinge_limit: Limits the hinge opening to given value
        """
        self.add_motion_goal(motion_goal_class=OpenDoorGoal.__name__,
                             tip_link=tip_link,
                             door_handle_link=door_handle_link,
                             name=name,
                             handle_limit=handle_limit,
                             hinge_limit=hinge_limit)

    def continuous_pointing_head(self):
        """
        Uses real_time_pointing for continuous tracking of a human_pose.
        """
        tip_V_pointing_axis: Vector3Stamped = Vector3Stamped()
        tip_V_pointing_axis.header.frame_id = 'head_center_camera_frame'
        tip_V_pointing_axis.vector.z = 1

        self.add_real_time_pointing(root_link='map',
                                    tip_link='head_center_camera_frame',
                                    topic_name='human_pose',
                                    pointing_axis=tip_V_pointing_axis)

    def add_adaptive_pouring(self,
                             root_link: str,
                             tip_link: str,
                             start_pose: PoseStamped,
                             tilt_angle: float,
                             tilt_axis: Vector3Stamped,
                             with_feedback: bool = True):
        """
        moves the tip_link to start_pose then tilts tip_link tip_angle amount around the tilt_axis.
        If with_feedback is True, then after tilting this goal will listen to the String topic /reasoner/concluded_behaviors
        and move tip_link in accordance with the motion primitives that are contained in the topic.
        The primitives are: increase, decrease moveForward, moveBack, moveLeft, moveRight, moveUp, moveDown, anticlockwise, clockwise.
        The goal will then finish after multiple messages containing {} are received.
        """
        self.add_motion_goal(motion_goal_class=PouringAdaptiveTilt.__name__,
                             root=root_link,
                             tip=tip_link,
                             pouring_pose=start_pose,
                             tilt_angle=tilt_angle,
                             tilt_axis=tilt_axis,
                             with_feedback=with_feedback)


class MonitorWrapper:
    _monitors: List[Monitor]
    max_trajectory_length_set: bool
    avoid_name_conflict: bool

    def __init__(self, robot_name: str, avoid_name_conflict: bool = False):
        self._robot_name = robot_name
        self.avoid_name_conflict = avoid_name_conflict
        self.max_trajectory_length_set = False
        self.reset()

    def get_monitors(self) -> List[Monitor]:
        return self._monitors

    def get_anded_monitor_names(self) -> str:
        non_cancel_monitors = []
        for monitor in self._monitors:
            if monitor.monitor_class not in get_all_classes_in_package('giskardpy.monitors', CancelMotion):
                non_cancel_monitors.append(f'\'{monitor.name}\'')
        return ' and '.join(non_cancel_monitors)

    def reset(self):
        self._monitors = []

    def add_monitor(self, *,
                    monitor_class: str,
                    name: Optional[str] = None,
                    start_condition: str = '',
                    **kwargs) -> str:
        """
        Generic function to add a monitor.
        :param monitor_class: Name of a class defined in src/giskardpy/monitors
        :param name: a unique name for the goal, will use class name by default
        :param start_condition: a logical expression to define the start condition for this monitor. e.g.
                                    not 'monitor1' and ('monitor2' or 'monitor3')
        :param kwargs: kwargs for __init__ function of motion_goal_class
        :return: the name of the monitor with added quotes to be used in logical expressions for conditions.
        """
        name = name or monitor_class
        if self.avoid_name_conflict:
            name = f'M{str(len(self._monitors))} {name}'
        if [x for x in self._monitors if x.name == name]:
            raise KeyError(f'monitor named {name} already exists.')
        monitor = giskard_msgs.Monitor()
        monitor.name = name
        monitor.monitor_class = monitor_class
        monitor.start_condition = start_condition
        monitor.kwargs = kwargs_to_json(kwargs)
        self._monitors.append(monitor)
        if not name.startswith('\'') and not name.startswith('"'):
            name = f'\'{name}\''  # put all monitor names in quotes so that the user doesn't have to
        return name

    def add_force_torque(self,
                         threshold_name: str,
                         object_type: str,
                         topic: str = '/filtered_raw/diff',
                         name: Optional[str] = None,
                         stay_true: bool = True,
                         start_condition: str = ''):
        """
        Can be used if planning wants to use their own grasping and placing actions, only adds force_torque_monitor
        without manipulations grasping/placing goals

        :param threshold_name: Name of the threshold to be used for the Force-Torque Monitor, options can be found in suturo_types.py
        :param object_type: Name of the object that is being placed, options can be found in suturo_types.py
        :param topic: name of the topic that the monitor should subscribe to, is hardcoded as '/filtered_raw/diff'
        :param name: name of the monitor, is optional, so can be left empty
        :param stay_true: whether the monitor should stay active until it is finished or not
        :param start_condition: condition for when the monitor should start checking for thresholds
        """
        return self.add_monitor(monitor_class=PayloadForceTorque.__name__,
                                name=name,
                                start_condition=start_condition,
                                stay_true=stay_true,
                                topic=topic,
                                threshold_name=threshold_name,
                                object_type=object_type)

    def add_local_minimum_reached(self,
                                  name: Optional[str] = None,
                                  stay_true: bool = True,
                                  start_condition: str = ''):
        """
        True if the world is currently in a local minimum.
        """
        return self.add_monitor(monitor_class=LocalMinimumReached.__name__,
                                name=name,
                                start_condition=start_condition,
                                stay_true=stay_true)

    def add_time_above(self,
                       threshold: float,
                       name: Optional[str] = None,
                       start_condition: str = ''):
        """
        True if the length of the trajectory is above threshold
        """
        return self.add_monitor(monitor_class=TimeAbove.__name__,
                                name=name,
                                start_condition=start_condition,
                                threshold=threshold)

    def add_joint_position(self,
                           goal_state: Dict[str, float],
                           threshold: float = 0.01,
                           name: Optional[str] = None,
                           start_condition: str = '',
                           stay_true: bool = True) -> str:
        """
        True if all joints in goal_state are closer than threshold to their respective value.
        """
        return self.add_monitor(monitor_class=JointGoalReached.__name__,
                                name=name,
                                goal_state=goal_state,
                                threshold=threshold,
                                start_condition=start_condition,
                                stay_true=stay_true)

    def add_cartesian_pose(self,
                           root_link: str,
                           tip_link: str,
                           goal_pose: PoseStamped,
                           root_group: Optional[str] = None,
                           tip_group: Optional[str] = None,
                           position_threshold: float = 0.01,
                           orientation_threshold: float = 0.01,
                           absolute: bool = False,
                           name: Optional[str] = None,
                           start_condition: str = '',
                           stay_true: bool = True):
        """
        True if tip_link is closer than the thresholds to goal_pose.
        """
        return self.add_monitor(monitor_class=PoseReached.__name__,
                                name=name,
                                root_link=root_link,
                                tip_link=tip_link,
                                goal_pose=goal_pose,
                                root_group=root_group,
                                tip_group=tip_group,
                                absolute=absolute,
                                start_condition=start_condition,
                                position_threshold=position_threshold,
                                orientation_threshold=orientation_threshold,
                                stay_true=stay_true)

    def add_cartesian_position(self,
                               root_link: str,
                               tip_link: str,
                               goal_point: PointStamped,
                               root_group: Optional[str] = None,
                               tip_group: Optional[str] = None,
                               threshold: float = 0.01,
                               absolute: bool = False,
                               name: Optional[str] = None,
                               start_condition: str = '',
                               stay_true: bool = True) -> str:
        """
        True if tip_link is closer than threshold to goal_point.
        """
        return self.add_monitor(monitor_class=PositionReached.__name__,
                                name=name,
                                root_link=root_link,
                                tip_link=tip_link,
                                goal_point=goal_point,
                                root_group=root_group,
                                start_condition=start_condition,
                                absolute=absolute,
                                tip_group=tip_group,
                                threshold=threshold,
                                stay_true=stay_true)

    def add_distance_to_line(self,
                             root_link: str,
                             tip_link: str,
                             center_point: PointStamped,
                             line_axis: Vector3Stamped,
                             line_length: float,
                             name: Optional[str] = None,
                             root_group: Optional[str] = None,
                             tip_group: Optional[str] = None,
                             stay_true: bool = True,
                             start_condition: str = '',
                             threshold: float = 0.01):
        """
        True if tip_link is closer than threshold to the line defined by center_point, line_axis and line_length.
        """
        return self.add_monitor(monitor_class=DistanceToLine.__name__,
                                name=name,
                                center_point=center_point,
                                line_axis=line_axis,
                                line_length=line_length,
                                root_link=root_link,
                                tip_link=tip_link,
                                start_condition=start_condition,
                                root_group=root_group,
                                stay_true=stay_true,
                                tip_group=tip_group,
                                threshold=threshold)

    def add_cartesian_orientation(self,
                                  root_link: str,
                                  tip_link: str,
                                  goal_orientation: QuaternionStamped,
                                  root_group: Optional[str] = None,
                                  tip_group: Optional[str] = None,
                                  threshold: float = 0.01,
                                  absolute: bool = False,
                                  name: Optional[str] = None,
                                  start_condition: str = '',
                                  stay_true: bool = True):
        """
        True if tip_link is closer than threshold to goal_orientation
        """
        return self.add_monitor(monitor_class=OrientationReached.__name__,
                                name=name,
                                root_link=root_link,
                                tip_link=tip_link,
                                goal_orientation=goal_orientation,
                                root_group=root_group,
                                tip_group=tip_group,
                                absolute=absolute,
                                start_condition=start_condition,
                                threshold=threshold,
                                stay_true=stay_true)

    def add_pointing_at(self,
                        goal_point: PointStamped,
                        tip_link: str,
                        pointing_axis: Vector3Stamped,
                        root_link: str,
                        name: Optional[str] = None,
                        tip_group: Optional[str] = None,
                        start_condition: str = '',
                        root_group: Optional[str] = None,
                        threshold: float = 0.01) -> str:
        """
        True if pointing_axis of tip_link is pointing at goal_point withing threshold.
        """
        return self.add_monitor(monitor_class=PointingAt.__name__,
                                name=name,
                                tip_link=tip_link,
                                goal_point=goal_point,
                                root_link=root_link,
                                tip_group=tip_group,
                                start_condition=start_condition,
                                root_group=root_group,
                                pointing_axis=pointing_axis,
                                threshold=threshold)

    def add_vectors_aligned(self,
                            root_link: str,
                            tip_link: str,
                            goal_normal: Vector3Stamped,
                            tip_normal: Vector3Stamped,
                            name: Optional[str] = None,
                            start_condition: str = '',
                            root_group: Optional[str] = None,
                            tip_group: Optional[str] = None,
                            threshold: float = 0.01) -> str:
        """
        True if tip_normal of tip_link is aligned with goal_normal within threshold.
        """
        return self.add_monitor(monitor_class=VectorsAligned.__name__,
                                name=name,
                                root_link=root_link,
                                tip_link=tip_link,
                                goal_normal=goal_normal,
                                tip_normal=tip_normal,
                                start_condition=start_condition,
                                root_group=root_group,
                                tip_group=tip_group,
                                threshold=threshold)

    def add_end_motion(self,
                       start_condition: str,
                       name: Optional[str] = None) -> str:
        """
        Ends the motion execution/planning if all start_condition are True.
        Use this to describe when your motion should end.
        """
        return self.add_monitor(monitor_class=EndMotion.__name__,
                                name=name,
                                start_condition=start_condition)

    def add_cancel_motion(self,
                          start_condition: str,
                          error_message: str,
                          error_code: int = GiskardError.ERROR,
                          name: Optional[str] = None) -> str:
        """
        Cancels the motion if all start_condition are True and will make Giskard return the specified error code.
        Use this to describe when failure conditions.
        """
        return self.add_monitor(monitor_class=CancelMotion.__name__,
                                name=name,
                                start_condition=start_condition,
                                error_message=error_message,
                                error_code=error_code)

    def add_max_trajectory_length(self,
                                  max_trajectory_length: Optional[float] = None) -> str:
        """
        A monitor that cancels the motion if the trajectory is longer than max_trajectory_length.
        """
        self.max_trajectory_length_set = True
        return self.add_monitor(name=None,
                                monitor_class=SetMaxTrajectoryLength.__name__,
                                new_length=max_trajectory_length,
                                start_condition='')

    def add_print(self,
                  message: str,
                  start_condition: str,
                  name: Optional[str] = None) -> str:
        """
        Debugging Monitor.
        Print a message to the terminal if all start_condition are True.
        """
        return self.add_monitor(monitor_class=Print.__name__,
                                name=name,
                                message=message,
                                start_condition=start_condition)

    def add_sleep(self,
                  seconds: float,
                  start_condition: str = '',
                  name: Optional[str] = None) -> str:
        """
        Calls rospy.sleep(seconds) when start_condition are True and turns True itself afterward.
        """
        return self.add_monitor(monitor_class=Sleep.__name__,
                                name=name,
                                seconds=seconds,
                                start_condition=start_condition)

    def add_set_seed_configuration(self,
                                   seed_configuration: Dict[str, float],
                                   group_name: Optional[str] = None,
                                   name: Optional[str] = None,
                                   start_condition: str = '') -> str:
        """
        Only meant for use with projection. Changes the world state to seed_configuration before starting planning,
        without having to plan a motion to it like with add_joint_position
        """
        return self.add_monitor(monitor_class=SetSeedConfiguration.__name__,
                                seed_configuration=seed_configuration,
                                group_name=group_name,
                                name=name,
                                start_condition=start_condition)

    def add_set_seed_odometry(self,
                              base_pose: PoseStamped,
                              group_name: Optional[str] = None,
                              name: Optional[str] = None,
                              start_condition: str = '') -> str:
        """
        Only meant for use with projection. Overwrites the odometry transform with base_pose.
        """
        return self.add_monitor(monitor_class=SetOdometry.__name__,
                                group_name=group_name,
                                base_pose=base_pose,
                                name=name,
                                start_condition=start_condition)

    def add_set_prediction_horizon(self, prediction_horizon: int, **kwargs: goal_parameter):
        """
        Will overwrite the prediction horizon for a single goal.
        Setting it to 1 will turn of acceleration and jerk limits.
        :param prediction_horizon: size of the prediction horizon, a number that should be 1 or above 5.
        """
        self.add_monitor(monitor_class=SetPredictionHorizon.__name__,
                         prediction_horizon=prediction_horizon,
                         **kwargs)

    def add_alternator(self,
                       start_condition: str = '',
                       name: Optional[str] = None,
                       mod: int = 2) -> str:
        """
        Testing monitor.
        True if floor(trajectory_length) % mod == 0.
        """
        if name is None:
            name = Alternator.__name__ + f' % {mod}'
        return self.add_monitor(monitor_class=Alternator.__name__,
                                name=name,
                                start_condition=start_condition,
                                mod=mod)

    def add_payload_alternator(self,
                               start_condition: str = '',
                               name: Optional[str] = None,
                               mod: int = 2) -> str:
        """
        Testing monitor.
        Like add_alternator but as a PayloadMonitor.
        """
        if name is None:
            name = PayloadAlternator.__name__ + f' % {mod}'
        return self.add_monitor(monitor_class=Alternator.__name__,
                                name=name,
                                start_condition=start_condition,
                                mod=mod)

    def add_payload_lidar(self,
                          start_condition: str = '',
                          topic: str = 'hsrb/base_scan',
                          name: Optional[str] = None,
                          frame_id: Optional[str] = 'base_range_sensor_link',
                          laser_distance_threshold_width: Optional[float] = 0.8,
                          laser_distance_threshold: Optional[float] = 0.5):
        """
        A monitor that detects points within a threshold of the robot via laser_scanner.
        Is True, when a point is detected within the threshold, False otherwise
        """
        if name is None:
            name = LidarPayloadMonitor.__name__ + f'_{topic}'
        return self.add_monitor(monitor_class=LidarPayloadMonitor.__name__,
                                name=name,
                                start_condition=start_condition,
                                topic=topic,
                                frame_id=frame_id,
                                laser_distance_threshold_width=laser_distance_threshold_width,
                                laser_distance_threshold=laser_distance_threshold)

    def add_open_hsr_gripper(self, start_condition: str = '', name: Optional[str] = None) -> str:
        """
        The monitor will send a force to the HSR's gripper to open it.
        """
        from giskardpy.monitors.hsr_gripper import OpenHsrGripper
        name = name or OpenHsrGripper.__name__
        return self.add_monitor(monitor_class=OpenHsrGripper.__name__,
                                name=name,
                                start_condition=start_condition)

    def add_close_hsr_gripper(self, start_condition: str = '', name: Optional[str] = None) -> str:
        """
        The monitor will send a force to the HSR's gripper to close it.
        """
        from giskardpy.monitors.hsr_gripper import CloseHsrGripper
        name = name or CloseHsrGripper.__name__
        return self.add_monitor(monitor_class=CloseHsrGripper.__name__,
                                name=name,
                                start_condition=start_condition)


class GiskardWrapper:
    last_feedback: MoveFeedback = None

    def __init__(self, node_name: str = 'giskard', avoid_name_conflict: bool = False, check_controller: bool = True):
        """
        Python wrapper for the ROS interface of Giskard.
        :param node_name: node name of Giskard
        :param avoid_name_conflict: if True, Giskard will automatically add an id to monitors and goals to avoid name
                                    conflicts.
        """
        self.list_controller_srv = None
        self.world = WorldWrapper(node_name)
        self.monitors = MonitorWrapper(self.robot_name, avoid_name_conflict=avoid_name_conflict)
        self.motion_goals = MotionGoalWrapper(self.robot_name, avoid_name_conflict=avoid_name_conflict)
        self.clear_motion_goals_and_monitors()
        giskard_topic = f'{node_name}/command'
        self._client = SimpleActionClient(giskard_topic, MoveAction)
        self._client.wait_for_server()
        self.clear_motion_goals_and_monitors()
        rospy.sleep(.3)

        if check_controller and self.world.get_control_mode() == ControlModes.close_loop:
            self.setup_controllers()

    def setup_controllers(self):
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

    def set_avoid_name_conflict(self, value: bool):
        self.avoid_name_conflict = value
        self.monitors.avoid_name_conflict = value
        self.motion_goals.avoid_name_conflict = value

    def add_default_end_motion_conditions(self) -> None:
        """
        1. Adds a local minimum reached monitor and adds it as an end_condition to all previously defined motion goals.
        2. Adds an end motion monitor, start_condition = all previously defined monitors are True.
        3. Adds a cancel motion monitor, start_condition = local minimum reached mit not all other monitors are True.
        4. Adds a max trajectory length monitor, if one wasn't added already.
        """
        local_min_reached_monitor_name = self.monitors.add_local_minimum_reached()
        for goal in self.motion_goals._goals:
            if goal.end_condition:
                goal.end_condition = f'({goal.end_condition}) and {local_min_reached_monitor_name}'
            else:
                goal.end_condition = local_min_reached_monitor_name
        self.monitors.add_end_motion(start_condition=self.monitors.get_anded_monitor_names())
        self.monitors.add_cancel_motion(start_condition=local_min_reached_monitor_name,
                                        error_message=f'local minimum reached',
                                        error_code=GiskardError.LOCAL_MINIMUM)
        if not self.monitors.max_trajectory_length_set:
            self.monitors.add_max_trajectory_length()
        self.monitors.max_trajectory_length_set = False

    def add_lidar_hold_condition(self) -> None:
        """
        1. Adds a lidar payload monitor and adds it as hold_condition to all previously defined motions goals
        """
        lidar_monitor_name = self.monitors.add_payload_lidar(laser_distance_threshold_width=0.3,
                                                             laser_distance_threshold=0.35)
        for goal in self.motion_goals._goals:
            if goal.hold_condition:
                goal.hold_condition = f'{goal.hold_condition} and {lidar_monitor_name}'
            else:
                goal.hold_condition = lidar_monitor_name

    @property
    def robot_name(self):
        return self.world.robot_name

    def clear_motion_goals_and_monitors(self):
        """
        Removes all move commands from the current goal, collision entries are left untouched.
        """
        self.motion_goals.reset()
        self.monitors.reset()

    def execute(self, wait: bool = True) -> MoveResult:
        """
        :param wait: this function blocks if wait=True
        :return: result from giskard
        """
        return self._send_action_goal(MoveGoal.EXECUTE, wait)

    def projection(self, wait: bool = True) -> MoveResult:
        """
        Plans, but doesn't execute the goal. Useful, if you just want to look at the planning ghost.
        :param wait: this function blocks if wait=True
        :return: result from Giskard
        """
        return self._send_action_goal(MoveGoal.PROJECTION, wait)

    def _send_action_goal(self, goal_type: int, wait: bool = True) -> Optional[MoveResult]:
        """
        Send goal to Giskard. Use this if you want to specify the goal_type, otherwise stick to wrappers like
        plan_and_execute.
        :param goal_type: one of the constants in MoveGoal
        :param wait: blocks if wait=True
        :return: result from Giskard
        """
        goal = self._create_action_goal()
        goal.type = goal_type
        if wait:
            self._client.send_goal_and_wait(goal)
            return self._client.get_result()
        else:
            self._client.send_goal(goal, feedback_cb=self._feedback_cb)

    def _create_action_goal(self) -> MoveGoal:
        action_goal = MoveGoal()
        action_goal.monitors = self.monitors.get_monitors()
        action_goal.goals = self.motion_goals.get_goals()
        self.clear_motion_goals_and_monitors()
        return action_goal

    def interrupt(self):
        """
        Stops the goal that was last sent to Giskard.
        """
        self._client.cancel_goal()

    def cancel_all_goals(self):
        """
        Stops any goal that Giskard is processing and attempts to halt the robot, even those not send from this client.
        """
        self._client.cancel_all_goals()

    def get_result(self, timeout: rospy.Duration = rospy.Duration()) -> MoveResult:
        """
        Waits for Giskard result and returns it. Only used when plan_and_execute was called with wait=False
        :param timeout: how long to wait
        """
        if not self._client.wait_for_result(timeout):
            raise TimeoutError('Timeout while waiting for goal.')
        return self._client.get_result()

    def _feedback_cb(self, msg: MoveFeedback):
        self.last_feedback = msg

    # SuTuRo Goals start here! (Only add SuTuRo goals which actually need monitors for stopping conditions etc.)
    def monitor_placing(self,
                        align: str,
                        grasp: str,
                        goal_pose: PoseStamped,
                        threshold_name: str = "",
                        object_type: str = "",
                        tip_link: str = 'hand_palm_link',
                        velocity: float = 0.02):
        """
        adds monitor functionality for the Placing motion goal, goal now stops if force_threshold is overstepped,
        which means the HSR essentially stops automatically after placing the object.

        :param align: alignment of action, should currently be either "vertical" or an empty string if not needed
        :param grasp: the direction from which the HSR should Grasp an object, in case of this method it should be direction the HSR is placing from
        :param goal_pose: where the object should be placed
        :param threshold_name: Name of the threshold to be used for the Force-Torque Monitor, options can be found in suturo_types.py
        :param object_type: Name of the object that is being placed, options can be found in suturo_types.py
        :param tip_link: name of the tip link, pre-defined as "hand_palm_link"
        :param velocity: the velocity that this action should be executed with
        """

        sleep = self.monitors.add_sleep(1.5)
        force_torque_trigger = self.monitors.add_monitor(monitor_class=PayloadForceTorque.__name__,
                                                         name=PayloadForceTorque.__name__,
                                                         start_condition='',
                                                         threshold_name=threshold_name,
                                                         object_type=object_type)

        self.motion_goals.add_motion_goal(motion_goal_class=Placing.__name__,
                                          goal_pose=goal_pose,
                                          align=align,
                                          grasp=grasp,
                                          tip_link=tip_link,
                                          velocity=velocity,
                                          end_condition=f'{force_torque_trigger} and {sleep}')

        local_min = self.monitors.add_local_minimum_reached()

        self.monitors.add_cancel_motion(local_min, "",
                                        GiskardError.FORCE_TORQUE_MONITOR_PLACING_MISSED_PLACING_LOCATION)
        self.monitors.add_end_motion(start_condition=force_torque_trigger)
        self.monitors.add_max_trajectory_length(100)

    def monitor_grasp_carefully(self,
                                goal_pose: PoseStamped,
                                align: str,
                                grasp: str,
                                reference_frame_alignment: Optional[str] = None,
                                object_name: str = "",
                                object_type: str = "",
                                threshold_name: str = "",
                                root_link: Optional[str] = None,
                                tip_link: Optional[str] = None):
        """
        adds monitor functionality to the reaching goal, thus making the original GraspCarefully motion goal redundant.
        The goal now stops if force_threshold/torque_threshold is undershot,
        which means it essentially stops automatically if the HSR for example slips off of a door handle while trying
        to open doors or fails to properly grip an object.

        :param goal_pose: where the object should be placed
        :param align: alignment of action, should currently be either "vertical" or an empty string if not needed
        :param grasp: the direction from which the HSR should Grasp an object
        :param reference_frame_alignment: reference frame to be used for alignment, can be left empty
        :param object_name: name of the object, needed for offset calculation of reaching goal
        :param object_type: Name of the object that is being placed, options can be found in suturo_types.py
        :param threshold_name: Name of the threshold to be used for the Force-Torque Monitor, options can be found in suturo_types.py
        :param root_link: root_link to be used, is optional, so should normally be left empty
        :param tip_link: name of the tip link, is optional, so can be left empty
        """
        sleep = self.monitors.add_sleep(1.5)
        # gripper_open = self.monitors.add_open_hsr_gripper()
        force_torque_trigger = self.monitors.add_monitor(monitor_class=PayloadForceTorque.__name__,
                                                         name=PayloadForceTorque.__name__,
                                                         start_condition='',
                                                         threshold_name=threshold_name,
                                                         object_type=object_type)

        self.motion_goals.add_motion_goal(motion_goal_class=Reaching.__name__,
                                          goal_pose=goal_pose,
                                          grasp=grasp,
                                          align=align,
                                          reference_frame_alignment=reference_frame_alignment,
                                          object_name=object_name,
                                          root_link=root_link,
                                          tip_link=tip_link,
                                          end_condition=f'{force_torque_trigger} and {sleep}')

        local_min = self.monitors.add_local_minimum_reached()

        self.monitors.add_cancel_motion(local_min, "", GiskardError.FORCE_TORQUE_MONITOR_GRASPING_MISSED_OBJECT)
        self.monitors.add_end_motion(start_condition=force_torque_trigger)
        self.monitors.add_max_trajectory_length(100)

    def monitor_transport_check(self,
                                object_type: str = "",
                                threshold_name: str = ""):
        """
        monitor for transportation of objects, currently WIP, but is supposed to register whether an object
        slips out of the HSRs grasp while transporting it (mostly useful for objects like cutlery, which are hard to grasp
        in a very reliant way)

        :param object_type: type of the object that is being transported, needed to determine correct threshold
        :param threshold_name: name of the motion, should be transport in this case, but the corresponding enum doesn't exist yet
        """

        gripper_closed = self.monitors.add_close_hsr_gripper()

        self.monitors.add_monitor(monitor_class=PayloadForceTorque.__name__,
                                  name=PayloadForceTorque.__name__,
                                  start_condition=gripper_closed,
                                  threshold_name=threshold_name,
                                  object_type=object_type)

    def monitor_full_grasping(self,
                              goal_pose: PoseStamped,
                              grasp: str,
                              align: str,
                              reference_frame_alignment: Optional[str] = None,
                              object_name: str = "",
                              object_type: str = "",
                              threshold_name: str = "",
                              root_link: Optional[str] = None,
                              tip_link: Optional[str] = None):
        """
        Similar to the grasping monitor, but with an added transport monitor at the end
        """

        self.monitor_grasp_carefully(goal_pose, grasp, align, reference_frame_alignment, object_name, object_type,
                                     threshold_name, root_link, tip_link)

        self.monitor_transport_check(object_type, threshold_name)

    # TODO: put logic into giskard Interface of Planning where Monitors and Motions are used
    # also other hsrb specific methods
    def grasp_bar_offset_goal(self,
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

    def hsrb_dishwasher_door_handle_grasp(self,
                                          handle_frame_id: str,
                                          grasp_bar_offset: float = 0.0,
                                          root_link: str = 'map',
                                          tip_link: str = 'hand_gripper_tool_frame'):
        """
        :param handle_frame_id: frame id of the dishwashers handle
        :param grasp_bar_offset: offset that is applied to the grasping axis
        :param root_link: root link, in this case the map link
        :param tip_link: tip link of the gripper, in this case the 'hand_gripper_tool_frame'
        """

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

        self.grasp_bar_offset_goal(root_link=root_link,
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

        self.motion_goals.add_align_planes(tip_link=tip_link,
                                           tip_normal=x_gripper,
                                           goal_normal=x_goal,
                                           root_link=root_link)

    def pre_pose_shelf_open(self,
                            offset_x: float = 0.03,
                            offset_y: float = 0.01,
                            offset_z: float = -0.15,
                            left_handle: str = 'shelf_hohc:shelf_door_left:handle',
                            left_door: str = 'shelf_hohc:shelf_door_left'):
        """
        Pre-Pose for opening the shelf

        :param offset_x: Depth offset for grasping pose of the handle
        :param offset_y: Width offset for grasping pose of the handle
        :param offset_z: Height offset for grasping pose of the handle
        :param left_handle: Left door handle of the shelf
        :param left_door: Main Link of the left door
        """
        if left_door not in self.world.get_group_names():
            self.world.register_group(new_group_name=left_door,
                                      root_link_group_name='suturo_shelf_hohc',
                                      root_link_name=left_door)

        first_goal = PoseStamped()
        first_goal.header.frame_id = left_handle
        first_goal.pose.position.x = offset_x
        first_goal.pose.position.y = offset_y
        first_goal.pose.position.z = offset_z
        first_goal.pose.orientation = Quaternion(*quaternion_from_matrix(np.array([[0, 1, 0, 0],
                                                                                   [0, 0, 1, 0],
                                                                                   [1, 0, 0, 0],
                                                                                   [0, 0, 0, 1]])))

        pre_grasp_reached = self.monitors.add_cartesian_pose(goal_pose=first_goal,
                                                             tip_link='hand_gripper_tool_frame',
                                                             root_link='map',
                                                             position_threshold=0.06)
        self.motion_goals.avoid_all_collisions(end_condition=pre_grasp_reached)
        self.motion_goals.allow_collision(group1='gripper', group2=left_door,
                                          start_condition=pre_grasp_reached)

        grasp_reached = self.monitors.add_cartesian_pose(goal_pose=first_goal,
                                                         tip_link='hand_gripper_tool_frame',
                                                         root_link='map')

        self.motion_goals.add_cartesian_pose(goal_pose=first_goal,
                                             tip_link='hand_gripper_tool_frame',
                                             root_link='map',
                                             weight=WEIGHT_BELOW_CA,
                                             end_condition=grasp_reached)

        self.monitors.add_end_motion(start_condition=grasp_reached)

    def open_shelf_door(self,
                        left_handle: str = 'shelf_hohc:shelf_door_left:handle',
                        left_door: str = 'shelf_hohc:shelf_door_left'):
        """
        Opens the shelf door.
        Requires the pre-pose to be reached and the gripper to be closed

        :param left_handle: Left door handle of the shelf
        :param left_door: Main Link of the left door
        """
        if left_door not in self.world.get_group_names():
            self.world.register_group(new_group_name=left_door,
                                      root_link_group_name='suturo_shelf_hohc',
                                      root_link_name=left_door)

        first_goal = PoseStamped()
        first_goal.header.frame_id = left_handle
        first_goal.pose.position.x = 0.03
        first_goal.pose.position.y = 0.01
        first_goal.pose.position.z = -0.15
        first_goal.pose.orientation = Quaternion(*quaternion_from_matrix(np.array([[0, 1, 0, 0],
                                                                                   [0, 0, 1, 0],
                                                                                   [1, 0, 0, 0],
                                                                                   [0, 0, 0, 1]])))

        pre_grasp_reached = self.monitors.add_cartesian_pose(goal_pose=first_goal,
                                                             tip_link='hand_gripper_tool_frame',
                                                             root_link='map',
                                                             position_threshold=0.06)
        self.motion_goals.avoid_all_collisions(end_condition=pre_grasp_reached)
        self.motion_goals.allow_collision(group1='gripper', group2=left_door,
                                          start_condition=pre_grasp_reached)

        self.motion_goals.add_open_container(tip_link='hand_gripper_tool_frame',
                                             environment_link=left_handle,
                                             goal_joint_state=-1.7,
                                             start_condition='')

        local_min = self.monitors.add_local_minimum_reached('done', start_condition='')
        self.monitors.add_end_motion(local_min)
