import rospy
from geometry_msgs.msg import PoseStamped, Quaternion

import giskardpy.casadi_wrapper
from giskardpy.suturo_types import GripperTypes
from manipulation_msgs.msg import ContextAlignVertical

from giskardpy.goals.suturo import GraspObject, ContextTypes, ContextActionModes
from giskardpy.python_interface.old_python_interface import OldGiskardWrapper
from giskardpy.utils import tfwrapper


class Demo:
    def __init__(self):
        rospy.init_node('testing')
        self.gis = OldGiskardWrapper()
        self.box_pose = PoseStamped()

    def drive(self):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.orientation.z = 1
        goal.pose.orientation.w = -0.04
        goal.pose.position.x = 1.70 # (+0.14)
        goal.pose.position.y = 1.75

        cart_mon = self.gis.monitors.add_cartesian_pose(goal_pose=goal,
                                                        tip_link='base_footprint',
                                                        root_link='map',
                                                        name='goal reached')

        self.gis.motion_goals.add_cartesian_pose(goal_pose=goal,
                                                 root_link='map',
                                                 tip_link='base_footprint',
                                                 hold_condition='',
                                                 end_condition=cart_mon)

        local_min = self.gis.monitors.add_local_minimum_reached(start_condition=cart_mon)

        self.gis.monitors.add_end_motion(start_condition=local_min)

        self.gis.execute()

    def drive2(self):
        goal2 = PoseStamped()
        goal2.header.frame_id = 'map'
        goal2.pose.orientation.z = 1
        goal2.pose.orientation.w = -0.04
        goal2.pose.position.x = 1.50  # (+0.14)
        goal2.pose.position.y = 1.75

        cart_mon2 = self.gis.monitors.add_cartesian_pose(goal_pose=goal2,
                                                         tip_link='base_footprint',
                                                         root_link='map',
                                                         name='goal reached')

        self.gis.motion_goals.add_cartesian_pose(goal_pose=goal2,
                                                 root_link='map',
                                                 tip_link='base_footprint',
                                                 hold_condition='',
                                                 end_condition=cart_mon2)

        local_min = self.gis.monitors.add_local_minimum_reached(start_condition=cart_mon2)

        self.gis.monitors.add_end_motion(start_condition=local_min)

        self.gis.execute()

    def pickup(self):
        self.gis.take_pose("pre_align_height")
        self.gis.execute(add_default=True)

        # align_pose = PoseStamped()
        # align_pose.header.frame_id = 'hand_gripper_tool_frame'
        # align_pose.pose.orientation.z = 1
        # align_pose.pose.orientation.w = -0.04
        # align_pose.pose.position.x = 1.70  # (+0.14)
        # align_pose.pose.position.y = 1.75
        # align_pose.pose.position.z = 1.2

        action = ContextTypes.context_action.value(content=ContextActionModes.grasping.value)
        from_above = ContextTypes.context_from_above.value(content=True)
        align_vertical = ContextAlignVertical()
        align_vertical.content = False
        shape = 'rectangle'

        context = {'action': action, 'from_above': from_above, 'align_vertical': align_vertical}

        self.gis.align_height(context=context, object_height=0.185, object_name=shape, goal_pose=self.box_pose)

        self.gis.execute(add_default=True)

        self.gis.change_gripper_state(GripperTypes.OPEN.value)

        self.gis.allow_collision('hsrb', 'cutlery')
        self.gis.reaching(context, object_name='cutlery', object_shape=shape, tip_link='hand_gripper_tool_frame')

        self.gis.execute(add_default=True)

        self.gis.allow_collision('hsrb', 'cutlery')
        self.gis.allow_collision('hsrb', 'iai_kitchen')
        self.gis.change_gripper_state(GripperTypes.CLOSE.value)

        self.gis.update_parent_link_of_group('cutlery', 'hand_gripper_tool_frame')

        test.drive()

        self.gis.align_height(context=context, object_height=0.185, object_name=shape, goal_pose=self.box_pose)

        self.gis.execute(add_default=True)

        goal_pose = self.box_pose
        goal_pose.pose.position.z -= 0.0004
        #goal_pose.pose.position.x += 0.20
        # goal_pose.pose.position.y += -0.70

        from_above = ContextTypes.context_from_above.value(content=True)
        align_vertical = ContextAlignVertical()
        align_vertical.content = False

        test.drive2()
        self.gis.allow_collision('hsrb', 'iai_kitchen')
        context = {'from_above': from_above, 'align_vertical': align_vertical}
        self.gis.placing(context, tip_link='hand_gripper_tool_frame', goal_pose=goal_pose, velocity=0.03)
        #self.gis.allow_all_collisions()
        self.gis.execute(add_default=True)

        self.gis.update_parent_link_of_group('cutlery', 'map')

        self.gis.allow_all_collisions()
        self.gis.change_gripper_state(GripperTypes.OPEN.value)

        self.gis.retract(object_name='cutlery', distance=0.25)
        self.gis.take_pose('park')

        self.gis.execute(add_default=True)

        self.gis.change_gripper_state(GripperTypes.NEUTRAL.value)

    def test(self):
        self.box_pose.header.frame_id = 'map'
        self.box_pose.pose.position.x = 1.00
        self.box_pose.pose.position.y = 1.75
        self.box_pose.pose.position.z = 0.746  # 0.726
        #self.box_pose.pose.orientation = Quaternion(0, 0, 1, 0)
        self.box_pose.pose.orientation = Quaternion(0, 0, 1, 1.3)

        #self.gis.add_box(name='milk', size=(0.085, 0.06, 0.195), pose=self.box_pose)
        self.gis.add_box(name='cutlery', size=(0.15, 0.021, 0.016), pose=self.box_pose)

    def add_kitchen(self):
        env_urdf = rospy.get_param('kitchen_description')
        kitchen_pose = tfwrapper.lookup_pose('map', 'iai_kitchen/urdf_main')
        self.gis.add_urdf(name='iai_kitchen',
                          urdf=env_urdf,
                          pose=kitchen_pose)


if __name__ == '__main__':
    test = Demo()

    test.drive()

    test.gis.remove_group('cutlery')
    test.gis.remove_group('bowl')
    test.gis.remove_group('plate')

    test.add_kitchen()

    test.test()
    test.pickup()
