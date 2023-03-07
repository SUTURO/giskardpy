from typing import Optional, Tuple, List

from geometry_msgs.msg import PoseStamped, PointStamped, Vector3, Vector3Stamped, Point

from giskardpy.goals.align_planes import AlignPlanes
from giskardpy.goals.cartesian_goals import CartesianPose, CartesianPositionStraight, CartesianPosition
from giskardpy.goals.goal import Goal
from giskardpy.goals.grasp_bar import GraspBar
from giskardpy.goals.joint_goals import JointPosition, JointPositionList
from giskardpy.goals.pointing import Pointing
from giskardpy.utils.logging import loginfo
from suturo_manipulation.gripper import Gripper
import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w


class SetBasePosition(Goal):
    def __init__(self):
        super().__init__()

        goal_new_state = {'arm_roll_joint': 1.6}
        self.add_constraints_of_goal(JointPositionList(goal_state=goal_new_state))

        loginfo(f'Moved hand out of sight')

    def make_constraints(self):
        pass

    def __str__(self) -> str:
        return super().__str__()


class MoveGripper(Goal):
    def __init__(self, open_gripper=1):
        super().__init__()
        g = Gripper(apply_force_action_server='/hsrb/gripper_controller/apply_force',
                    follow_joint_trajectory_server='/hsrb/gripper_controller/follow_joint_trajectory')

        if open_gripper == 1:
            g.set_gripper_joint_position(1)
        else:
            g.close_gripper_force(1)

    def make_constraints(self):
        pass

    def __str__(self) -> str:
        return super().__str__()


class PrepareGraspBox(Goal):
    def __init__(self,
                 box_pose: PoseStamped,
                 box_size: Tuple[float],
                 root_link: Optional[str] = 'map',
                 tip_link: Optional[str] = 'hand_palm_link',
                 wrist_flex=-1.57,
                 wrist_roll=0.0
                 ):
        """
        Move to a given position where a box can be grasped.

        :param box_pose: center position of the grasped object
        :param tip_link: name of the tip link of the kin chain
        :param box_z_length: length of the box along the z-axis
        :param box_x_length: length of the box along the x-axis
        :param box_y_length: length of the box along the y-axis

        """
        super().__init__()
        # giskard_link_name = self.world.get_link_name(tip_link)
        # root_link = self.world.root_link_name
        # map_box_pose = self.transform_msg('map', box_pose)

        self.wrist_roll = wrist_roll

        box_point = PointStamped()
        box_point.header.frame_id = root_link
        box_point.point.x = box_pose.pose.position.x
        box_point.point.y = box_pose.pose.position.y
        box_point.point.z = box_pose.pose.position.z

        # root link
        self.root = self.world.get_link_name(root_link, None)
        self.tip = self.world.get_link_name(tip_link, None)
        self.root_P_goal_point = self.transform_msg(self.root, box_point)

        # tip link
        giskard_link_name = str(self.world.get_link_name(tip_link))
        loginfo('giskard_link_name: {}'.format(giskard_link_name))

        # tip_axis
        tip_grasp_a = Vector3Stamped()
        tip_grasp_a.header.frame_id = giskard_link_name

        box_size_array = [box_size[0], box_size[1], box_size[2]]

        # tip_grasp_a.vector = set_grasp_axis(box_size_array)

        tip_grasp_a.vector.x = 1
        # print(tip_grasp_a.vector)

        # if grasp_type == 1:
        #   tip_grasp_a.vector.x = 1
        #   box_pose.pose.position.y -= 0.078
        # else:
        #    tip_grasp_a.vector.y = 1
        #    box_pose.pose.position.y += 0.078

        # bar_center
        bar_c = PointStamped()
        bar_c.point = box_pose.pose.position

        # bar_axis
        bar_a = Vector3Stamped()
        bar_a.header.frame_id = root_link
        bar_a.vector = set_grasp_axis(box_size_array, minimum=False)
        # bar_a.vector.z = 1

        tolerance = 0.8
        bar_l = max(box_size_array) * tolerance

        # align with axis
        tip_grasp_axis_b = Vector3Stamped()
        tip_grasp_axis_b.header.frame_id = giskard_link_name

        tip_grasp_axis_b.vector.y = 1
        # else:
        #    tip_grasp_axis_b.vector.z = -1

        bar_axis_b = Vector3Stamped()
        bar_axis_b.header.frame_id = root_link
        bar_axis_b.vector.x = 1


        self.add_constraints_of_goal(Pointing(root_link=root_link,
                                              tip_link=giskard_link_name,
                                              goal_point=box_point))

        
        '''                          
        self.add_constraints_of_goal(AlignPlanes(root_link=root_link,
                                                 tip_link=giskard_link_name,
                                                 goal_normal=bar_axis_b,
                                                 tip_normal=tip_grasp_axis_b))
        '''
        # align hand with z

        print(bar_a)
        print(tip_grasp_a)

        self.add_constraints_of_goal(GraspBar(root_link=root_link,
                                              tip_link=giskard_link_name,
                                              tip_grasp_axis=tip_grasp_a,
                                              bar_center=bar_c,
                                              bar_axis=bar_a,
                                              bar_length=bar_l))

        self.add_constraints_of_goal(MoveGripper(True))


        '''
        w_roll values:

        grasp vertical: 1.6

        grasp horizontal: 0.5
        '''
        goal_state = {
            #u'wrist_flex_joint': wrist_flex,
            u'wrist_roll_joint': wrist_roll}
        #hard = True
        #self.add_constraints_of_goal(JointPositionList(
        #    goal_state=goal_state))

    def make_constraints(self):
        '''
        goal_state = {
            #u'wrist_flex_joint': wrist_flex,
            u'wrist_roll_joint': self.wrist_roll}
        #hard = True
        self.add_constraints_of_goal(JointPositionList(
            goal_state=goal_state))
        '''

    def __str__(self) -> str:
        return super().__str__()


class MoveDrawer(Goal):
    def __init__(self,
                 knob_pose: PoseStamped,
                 direction: Optional[Vector3] = None,
                 distance: Optional[float] = None,
                 open_drawer: Optional[int] = 1,
                 align_horizontal: Optional[int] = 1):
        """
        Move drawer in a given direction.
        The drawer knob has to be grasped first, f.e. with PrepareGraspBox.
        :param knob_pose: current position of the knob
        :param direction: direction vector in which the drawer should move
        :param distance: distance the drawer should move
        """

        super().__init__()

        if direction is None:
            direction = Vector3()
            if open_drawer == 1:
                direction.y = 1
            else:
                direction.y = -1

        if distance is None:
            distance = 0.4  # mueslibox
            # distance = 0.4 - 0.075  # drawer

        root_l = 'map'
        giskard_link_name = str(self.world.get_link_name('hand_palm_link'))

        new_x = (direction.x * distance) + knob_pose.pose.position.x
        new_y = (direction.y * distance) + knob_pose.pose.position.y
        new_z = (direction.z * distance) + knob_pose.pose.position.z
        calculated_position = Vector3(new_x, new_y, new_z)

        goal_position = PoseStamped()
        goal_position.header.frame_id = root_l
        goal_position.pose.position = calculated_position
        loginfo(goal_position)

        # position straight
        goal_pos = PoseStamped()
        goal_pos.header.frame_id = root_l
        goal_pos.pose.position = calculated_position

        tip_grasp_axis_b = Vector3Stamped()
        tip_grasp_axis_b.header.frame_id = giskard_link_name

        bar_axis_b = Vector3Stamped()

        if align_horizontal == 1:
            tip_grasp_axis_b.vector.x = 1
            bar_axis_b.vector.z = 1
        else:
            tip_grasp_axis_b.vector.z = -1
            bar_axis_b.vector.y = 1

        self.add_constraints_of_goal(AlignPlanes(root_link=root_l,
                                                 tip_link=giskard_link_name,
                                                 goal_normal=bar_axis_b,
                                                 tip_normal=tip_grasp_axis_b))

        self.add_constraints_of_goal(CartesianPose(root_link=root_l,
                                                   tip_link=giskard_link_name,
                                                   goal_pose=goal_pos))

    def make_constraints(self):
        pass

    def __str__(self) -> str:
        return super().__str__()


class PlaceObject(Goal):
    def __init__(self,
                 goal_pose: PoseStamped,
                 object_height: Optional[float] = 0.1):
        super().__init__()

        root_l = 'map'
        giskard_link_name = str(self.world.get_link_name('hand_palm_link'))

        goal_pose.pose.position.z = goal_pose.pose.position.z + (object_height / 2)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = 'map'
        bar_axis.vector.y = 1

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = giskard_link_name
        tip_grasp_axis.vector.z = 1

        bar_axis_b = Vector3Stamped()
        bar_axis_b.header.frame_id = 'map'
        bar_axis_b.vector.x = 1

        tip_grasp_axis_b = Vector3Stamped()
        tip_grasp_axis_b.header.frame_id = giskard_link_name
        tip_grasp_axis_b.vector.y = 1

        self.add_constraints_of_goal(AlignPlanes(root_link=root_l,
                                                 tip_link=giskard_link_name,
                                                 goal_normal=bar_axis,
                                                 tip_normal=tip_grasp_axis))

        self.add_constraints_of_goal(AlignPlanes(root_link=root_l,
                                                 tip_link=giskard_link_name,
                                                 goal_normal=bar_axis_b,
                                                 tip_normal=tip_grasp_axis_b))

        self.add_constraints_of_goal(CartesianPose(root_link=root_l,
                                                   tip_link=giskard_link_name,
                                                   goal_pose=goal_pose))

    def make_constraints(self):
        pass

    def __str__(self) -> str:
        return super().__str__()


def set_grasp_axis(axes: List[float],
                   minimum: Optional[bool] = True):
    values = axes
    values.sort(reverse=minimum)
    index_sorted_values = []
    for e in values:
        index_sorted_values.append(values.index(e))


    grasp_vector = Vector3()
    if index_sorted_values[0] == 0:
        grasp_vector.x = 1
    elif index_sorted_values[0] == 1:
        grasp_vector.y = 1
    else:
        grasp_vector.z = 1

    return grasp_vector
