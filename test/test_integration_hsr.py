import os
from copy import deepcopy
from typing import Optional

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped, Vector3Stamped
from numpy import pi
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

from giskard_msgs.msg import GiskardError
from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.hsr import HSRCollisionAvoidanceConfig, WorldWithHSRConfig, HSRStandaloneInterface
from giskardpy.configs.qp_controller_config import QPControllerConfig
from giskardpy.god_map import god_map
from giskardpy.monitors.force_monitor import Payload_Force
from giskardpy.monitors.lidar_monitor import LidarPayloadMonitor
from giskardpy.python_interface.old_python_interface import OldGiskardWrapper
from giskardpy.suturo_types import ForceTorqueThresholds
from giskardpy.utils.utils import launch_launchfile
from utils_for_tests import compare_poses, GiskardTestWrapper

if 'GITHUB_WORKFLOW' not in os.environ:
    from giskardpy.goals.suturo import ContextActionModes, ContextTypes


class HSRTestWrapper(GiskardTestWrapper):
    default_pose = {
        'arm_flex_joint': -0.03,
        'arm_lift_joint': 0.01,
        'arm_roll_joint': 0.0,
        'head_pan_joint': 0.0,
        'head_tilt_joint': 0.0,
        'wrist_flex_joint': 0.0,
        'wrist_roll_joint': 0.0,
    }
    better_pose = default_pose

    def __init__(self, giskard=None):
        self.tip = 'hand_gripper_tool_frame'
        if giskard is None:
            giskard = Giskard(world_config=WorldWithHSRConfig(),
                              collision_avoidance_config=HSRCollisionAvoidanceConfig(),
                              robot_interface_config=HSRStandaloneInterface(),
                              behavior_tree_config=StandAloneBTConfig(debug_mode=True, publish_js=True),
                              qp_controller_config=QPControllerConfig())
        super().__init__(giskard)
        self.gripper_group = 'gripper'
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.odom_root = 'odom'
        self.robot = god_map.world.groups[self.robot_name]

    def low_level_interface(self):
        return super(OldGiskardWrapper, self)

    def move_base(self, goal_pose):
        self.set_cart_goal(goal_pose, tip_link='base_footprint', root_link=god_map.world.root_link_name)
        self.plan_and_execute()

    def open_gripper(self):
        self.command_gripper(1.23)

    def close_gripper(self):
        self.command_gripper(0)

    def command_gripper(self, width):
        js = {'hand_motor_joint': width}
        self.set_joint_goal(js)
        self.plan_and_execute()

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        if god_map.is_standalone():
            self.teleport_base(p)
        else:
            self.move_base(p)

    def reset(self):
        self.clear_world()
        # self.close_gripper()
        self.reset_base()
        self.register_group('gripper',
                            root_link_group_name=self.robot_name,
                            root_link_name='hand_palm_link')

    def teleport_base(self, goal_pose, group_name: Optional[str] = None):
        self.set_seed_odometry(base_pose=goal_pose, group_name=group_name)
        self.allow_all_collisions()
        self.plan_and_execute()


@pytest.fixture(scope='module')
def giskard(request, ros):
    launch_launchfile('package://hsr_description/launch/upload_hsrb.launch')
    c = HSRTestWrapper()
    # c = HSRTestWrapperMujoco()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def box_setup(zero_pose: HSRTestWrapper) -> HSRTestWrapper:
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.1
    p.pose.orientation.w = 1
    zero_pose.add_box_to_world(name='box', size=(1, 1, 1), pose=p)
    return zero_pose


# TODO: Further rework force Monitor test; removing unnecessary Code, create more Tests etc.
class TestForceMonitor:

    def test_force_monitor_placing(self, zero_pose: HSRTestWrapper):
        sleep = zero_pose.monitors.add_sleep(2.5)
        force_torque = zero_pose.monitors.add_monitor(monitor_class=Payload_Force.__name__,
                                                      name=Payload_Force.__name__,
                                                      start_condition='',
                                                      threshold_name=ForceTorqueThresholds.FT_Placing.value)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation.w = 1
        goal_reached = zero_pose.monitors.add_cartesian_pose(goal_pose=base_goal,
                                                             tip_link='base_footprint',
                                                             root_link='map',
                                                             name='goal reached')

        zero_pose.motion_goals.add_cartesian_pose(goal_pose=base_goal,
                                                  tip_link='base_footprint',
                                                  root_link='map',
                                                  hold_condition=force_torque,
                                                  end_condition=f'{goal_reached} and {sleep}')
        local_min = zero_pose.monitors.add_local_minimum_reached(start_condition=goal_reached)

        end = zero_pose.monitors.add_end_motion(start_condition=f'{local_min} and {sleep}')
        zero_pose.motion_goals.allow_all_collisions()
        zero_pose.set_max_traj_length(100)
        zero_pose.execute(add_local_minimum_reached=False)

    def test_force_monitor_grasp(self, zero_pose: HSRTestWrapper):
        sleep = zero_pose.monitors.add_sleep(2.5)
        force_torque = zero_pose.monitors.add_monitor(monitor_class=Payload_Force.__name__,
                                                      name=Payload_Force.__name__,
                                                      start_condition='',
                                                      threshold_name=ForceTorqueThresholds.FT_GraspWithCare.value)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation.w = 1
        goal_reached = zero_pose.monitors.add_cartesian_pose(goal_pose=base_goal,
                                                             tip_link='base_footprint',
                                                             root_link='map',
                                                             name='goal reached')

        zero_pose.motion_goals.add_cartesian_pose(goal_pose=base_goal,
                                                  tip_link='base_footprint',
                                                  root_link='map',
                                                  hold_condition=force_torque,
                                                  end_condition=f'{goal_reached} and {sleep}')
        local_min = zero_pose.monitors.add_local_minimum_reached(start_condition=goal_reached)

        end = zero_pose.monitors.add_end_motion(start_condition=f'{local_min} and {sleep}')
        zero_pose.motion_goals.allow_all_collisions()
        zero_pose.set_max_traj_length(100)
        zero_pose.execute(add_local_minimum_reached=False)

class TestLidarMonitor:

    # Zur Zeit kein automatisch ausführbarer Test
    def test_lidar_monitor(self, zero_pose: HSRTestWrapper):
        lidar = zero_pose.monitors.add_monitor(monitor_class=LidarPayloadMonitor.__name__,
                                               name=LidarPayloadMonitor.__name__ + 'Test',
                                               start_condition='',
                                               topic='/hokuyo_back/most_intense',
                                               frame_id='laser_reference_back',
                                               laser_distance_threshold_width=0.5,
                                               laser_distance_threshold=0.8)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation.w = 1
        goal_reached = zero_pose.monitors.add_cartesian_pose(goal_pose=base_goal,
                                                             tip_link='base_footprint',
                                                             root_link='map',
                                                             name='goal reached')

        zero_pose.motion_goals.add_cartesian_pose(goal_pose=base_goal,
                                                  tip_link='base_footprint',
                                                  root_link='map',
                                                  hold_condition=lidar,
                                                  end_condition=f'{goal_reached}')

        local_min = zero_pose.monitors.add_local_minimum_reached(start_condition=goal_reached)

        zero_pose.monitors.add_end_motion(start_condition=f'{local_min}')
        zero_pose.motion_goals.allow_all_collisions()
        zero_pose.execute(add_local_minimum_reached=False)


class TestJointGoals:

    def test_mimic_joints(self, zero_pose: HSRTestWrapper):
        arm_lift_joint = god_map.world.search_for_joint_name('arm_lift_joint')
        zero_pose.open_gripper()
        hand_T_finger_current = god_map.world.compute_fk_pose('hand_palm_link', 'hand_l_distal_link')
        hand_T_finger_expected = PoseStamped()
        hand_T_finger_expected.header.frame_id = 'hand_palm_link'
        hand_T_finger_expected.pose.position.x = -0.01675
        hand_T_finger_expected.pose.position.y = -0.0907
        hand_T_finger_expected.pose.position.z = 0.0052
        hand_T_finger_expected.pose.orientation.x = -0.0434
        hand_T_finger_expected.pose.orientation.y = 0.0
        hand_T_finger_expected.pose.orientation.z = 0.0
        hand_T_finger_expected.pose.orientation.w = 0.999
        compare_poses(hand_T_finger_current.pose, hand_T_finger_expected.pose)

        js = {'torso_lift_joint': 0.1}
        zero_pose.set_joint_goal(js, add_monitor=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(god_map.world.state[arm_lift_joint].position, 0.2, decimal=2)
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = 'base_footprint'
        base_T_torso.pose.position.x = 0
        base_T_torso.pose.position.y = 0
        base_T_torso.pose.position.z = 0.8518
        base_T_torso.pose.orientation.x = 0
        base_T_torso.pose.orientation.y = 0
        base_T_torso.pose.orientation.z = 0
        base_T_torso.pose.orientation.w = 1
        base_T_torso2 = god_map.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

        zero_pose.close_gripper()

    def test_mimic_joints2(self, zero_pose: HSRTestWrapper):
        arm_lift_joint = god_map.world.search_for_joint_name('arm_lift_joint')
        zero_pose.open_gripper()

        tip = 'hand_gripper_tool_frame'
        p = PoseStamped()
        p.header.frame_id = tip
        p.pose.position.z = 0.2
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=p, tip_link=tip,
                                root_link='base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(god_map.world.state[arm_lift_joint].position, 0.2, decimal=2)
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = 'base_footprint'
        base_T_torso.pose.position.x = 0
        base_T_torso.pose.position.y = 0
        base_T_torso.pose.position.z = 0.8518
        base_T_torso.pose.orientation.x = 0
        base_T_torso.pose.orientation.y = 0
        base_T_torso.pose.orientation.z = 0
        base_T_torso.pose.orientation.w = 1
        base_T_torso2 = god_map.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

        zero_pose.close_gripper()

    def test_mimic_joints3(self, zero_pose: HSRTestWrapper):
        arm_lift_joint = god_map.world.search_for_joint_name('arm_lift_joint')
        zero_pose.open_gripper()
        tip = 'head_pan_link'
        p = PoseStamped()
        p.header.frame_id = tip
        p.pose.position.z = 0.15
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=p, tip_link=tip,
                                root_link='base_footprint')
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(god_map.world.state[arm_lift_joint].position, 0.3, decimal=2)
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = 'base_footprint'
        base_T_torso.pose.position.x = 0
        base_T_torso.pose.position.y = 0
        base_T_torso.pose.position.z = 0.902
        base_T_torso.pose.orientation.x = 0
        base_T_torso.pose.orientation.y = 0
        base_T_torso.pose.orientation.z = 0
        base_T_torso.pose.orientation.w = 1
        base_T_torso2 = god_map.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

        zero_pose.close_gripper()

    def test_mimic_joints4(self, zero_pose: HSRTestWrapper):
        ll, ul = god_map.world.get_joint_velocity_limits('hsrb/arm_lift_joint')
        assert ll == -0.15
        assert ul == 0.15
        ll, ul = god_map.world.get_joint_velocity_limits('hsrb/torso_lift_joint')
        assert ll == -0.075
        assert ul == 0.075
        joint_goal = {'torso_lift_joint': 0.25}
        zero_pose.set_joint_goal(joint_goal, add_monitor=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(god_map.world.state['hsrb/arm_lift_joint'].position, 0.5, decimal=2)


class TestCartGoals:
    def test_save_graph_pdf(self, kitchen_setup):
        box1_name = 'box1'
        pose = PoseStamped()
        pose.header.frame_id = kitchen_setup.default_root
        pose.pose.orientation.w = 1
        kitchen_setup.add_box_to_world(name=box1_name,
                                       size=(1, 1, 1),
                                       pose=pose,
                                       parent_link='hand_palm_link',
                                       parent_link_group='hsrb')
        god_map.world.save_graph_pdf()

    def test_move_base(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.teleport_base(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.set_cart_goal(goal_pose=base_goal, tip_link='base_footprint', root_link='map')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_move_base_1m_forward(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.move_base(map_T_odom)

    def test_move_base_1m_left(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.move_base(map_T_odom)

    def test_move_base_1m_diagonal(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.move_base(map_T_odom)

    def test_move_base_rotate(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(map_T_odom)

    def test_move_base_forward_rotate(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(map_T_odom)

    def test_rotate_gripper(self, zero_pose: HSRTestWrapper):
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.tip
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.set_cart_goal(goal_pose=r_goal, tip_link=zero_pose.tip, root_link='map')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()


# FIXME ForceSensorGoal needs to be fixed, before the Tests can be fixed
# class TestConstraints:
#
#     def test_open_fridge(self, kitchen_setup: HSRTestWrapper):
#         handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'
#         handle_name = 'iai_fridge_door_handle'
#         kitchen_setup.open_gripper()
#         base_goal = PoseStamped()
#         base_goal.header.frame_id = 'map'
#         base_goal.pose.position = Point(0.3, -0.5, 0)
#         base_goal.pose.orientation.w = 1
#         kitchen_setup.move_base(base_goal)
#
#         bar_axis = Vector3Stamped()
#         bar_axis.header.frame_id = handle_frame_id
#         bar_axis.vector.z = 1
#
#         bar_center = PointStamped()
#         bar_center.header.frame_id = handle_frame_id
#
#         tip_grasp_axis = Vector3Stamped()
#         tip_grasp_axis.header.frame_id = kitchen_setup.tip
#         tip_grasp_axis.vector.x = 1
#
#         kitchen_setup.set_grasp_bar_goal(root_link=kitchen_setup.default_root,
#                                          tip_link=kitchen_setup.tip,
#                                          tip_grasp_axis=tip_grasp_axis,
#                                          bar_center=bar_center,
#                                          bar_axis=bar_axis,
#                                          bar_length=.4)
#         x_gripper = Vector3Stamped()
#         x_gripper.header.frame_id = kitchen_setup.tip
#         x_gripper.vector.z = 1
#
#         x_goal = Vector3Stamped()
#         x_goal.header.frame_id = handle_frame_id
#         x_goal.vector.x = -1
#         kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.tip,
#                                             tip_normal=x_gripper,
#                                             goal_normal=x_goal,
#                                             root_link='map')
#         kitchen_setup.allow_all_collisions()
#         # kitchen_setup.add_json_goal('AvoidJointLimits', percentage=10)
#         kitchen_setup.execute()
#         current_pose = god_map.world.compute_fk_pose(root='map', tip=kitchen_setup.tip)
#
#         kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.tip,
#                                               environment_link=handle_name,
#                                               goal_joint_state=1.5)
#         # kitchen_setup.motion_goals.add_motion_goal('AvoidJointLimits', percentage=40)
#         kitchen_setup.allow_all_collisions()
#         # kitchen_setup.add_json_goal('AvoidJointLimits')
#         kitchen_setup.execute()
#         kitchen_setup.set_env_state({'iai_fridge_door_joint': 1.5})
#
#         pose_reached = kitchen_setup.monitors.add_cartesian_pose('map',
#                                                                  tip_link=kitchen_setup.tip,
#                                                                  goal_pose=current_pose)
#         kitchen_setup.monitors.add_end_motion(start_condition=pose_reached)
#
#         kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.tip,
#                                               environment_link=handle_name,
#                                               goal_joint_state=0)
#         kitchen_setup.allow_all_collisions()
#         # kitchen_setup.motion_goals.add_motion_goal('AvoidJointLimits', percentage=40)
#
#         kitchen_setup.execute(add_local_minimum_reached=False)
#
#         kitchen_setup.set_env_state({'iai_fridge_door_joint': 0})
#
#         kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
#         kitchen_setup.allow_self_collision()
#         kitchen_setup.plan_and_execute()
#
#         kitchen_setup.close_gripper()


class TestCollisionAvoidanceGoals:

    def test_self_collision_avoidance_empty(self, zero_pose: HSRTestWrapper):
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_code=GiskardError.EMPTY_PROBLEM)
        current_state = god_map.world.state.to_position_dict()
        current_state = {k.short_name: v for k, v in current_state.items()}
        zero_pose.compare_joint_state(current_state, zero_pose.default_pose)

    def test_self_collision_avoidance(self, zero_pose: HSRTestWrapper):
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.tip
        r_goal.pose.position.z = 0.5
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=r_goal, tip_link=zero_pose.tip, root_link='map')
        zero_pose.plan_and_execute()

    def test_self_collision_avoidance2(self, zero_pose: HSRTestWrapper):
        js = {
            'arm_flex_joint': -0.03,
            'arm_lift_joint': 0.0,
            'arm_roll_joint': -1.52,
            'head_pan_joint': -0.09,
            'head_tilt_joint': -0.62,
            'wrist_flex_joint': -1.55,
            'wrist_roll_joint': 0.11,
        }
        zero_pose.set_seed_configuration(js)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'hand_palm_link'
        goal_pose.pose.position.x = 0.5
        goal_pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=goal_pose, tip_link=zero_pose.tip, root_link='map')
        zero_pose.plan_and_execute()

    def test_attached_collision1(self, box_setup: HSRTestWrapper):
        box_name = 'asdf'
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'map'
        box_pose.pose.position = Point(0.85, 0.3, .66)
        box_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        box_setup.add_box_to_world(box_name, (0.07, 0.04, 0.1), box_pose)
        box_setup.open_gripper()

        grasp_pose = deepcopy(box_pose)
        # grasp_pose.pose.position.x -= 0.05
        grasp_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                          [0, -1, 0, 0],
                                                                          [1, 0, 0, 0],
                                                                          [0, 0, 0, 1]]))
        box_setup.set_cart_goal(goal_pose=grasp_pose, tip_link=box_setup.tip, root_link='map')
        box_setup.plan_and_execute()
        box_setup.update_parent_link_of_group(box_name, box_setup.tip)

        base_goal = PoseStamped()
        base_goal.header.frame_id = box_setup.default_root
        base_goal.pose.position.x -= 0.5
        base_goal.pose.orientation.w = 1
        box_setup.move_base(base_goal)

        box_setup.close_gripper()

    def test_collision_avoidance(self, zero_pose: HSRTestWrapper):
        js = {'arm_flex_joint': -np.pi / 2}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 0.9
        p.pose.position.y = 0
        p.pose.position.z = 0.5
        p.pose.orientation.w = 1
        zero_pose.add_box_to_world(name='box', size=(1, 1, 0.01), pose=p)

        js = {'arm_flex_joint': 0}
        zero_pose.set_joint_goal(js, add_monitor=False)
        zero_pose.plan_and_execute()


class TestAddObject:
    def test_add(self, zero_pose):
        box1_name = 'box1'
        pose = PoseStamped()
        pose.header.frame_id = zero_pose.default_root
        pose.pose.orientation.w = 1
        pose.pose.position.x = 1
        zero_pose.add_box_to_world(name=box1_name,
                                   size=(1, 1, 1),
                                   pose=pose,
                                   parent_link='hand_palm_link')

        zero_pose.set_joint_goal({'arm_flex_joint': -0.7})
        zero_pose.plan_and_execute()


class TestSUTURO:

    # TODO: Actually make the reach Test test something
    def test_reaching(self, zero_pose: HSRTestWrapper):
        pass

    def test_grasp_object(self, zero_pose: HSRTestWrapper):
        from_above_modes = [False, True]
        align_vertical_modes = [False, True]

        grasp_pose_1 = PoseStamped()
        grasp_pose_1.header.frame_id = 'map'
        grasp_pose_1.pose.position.x = 1.0701112670482553
        grasp_pose_1.pose.position.y = 0.0001316214338790437
        grasp_pose_1.pose.position.z = 0.6900203701423123
        grasp_pose_1.pose.orientation.x = 0.7071167396552901
        grasp_pose_1.pose.orientation.y = 6.533426136710821e-05
        grasp_pose_1.pose.orientation.z = 0.7070968173260486
        grasp_pose_1.pose.orientation.w = -5.619679601192823e-05

        grasp_pose_2 = PoseStamped()
        grasp_pose_2.header.frame_id = 'map'
        grasp_pose_2.pose.position.x = 1.0699999955917536
        grasp_pose_2.pose.position.y = 1.1264868679568747e-05
        grasp_pose_2.pose.position.z = 0.6900040048607661
        grasp_pose_2.pose.orientation.x = -0.49989060882740377
        grasp_pose_2.pose.orientation.y = 0.50010930896272
        grasp_pose_2.pose.orientation.z = -0.49972084003542927
        grasp_pose_2.pose.orientation.w = 0.5002790624534303

        grasp_pose_3 = PoseStamped()
        grasp_pose_3.header.frame_id = 'map'
        grasp_pose_3.pose.position.x = 1.0000098440969631
        grasp_pose_3.pose.position.y = -5.5826046789126e-07
        grasp_pose_3.pose.position.z = 0.6299900562082916
        grasp_pose_3.pose.orientation.x = 0.9999997719507597
        grasp_pose_3.pose.orientation.y = 0.0006753489230108196
        grasp_pose_3.pose.orientation.z = -1.064646728699401e-06
        grasp_pose_3.pose.orientation.w = -1.0617940975076199e-06

        grasp_pose_4 = PoseStamped()
        grasp_pose_4.header.frame_id = 'map'
        grasp_pose_4.pose.position.x = 0.9999945694917095
        grasp_pose_4.pose.position.y = 4.234772015936794e-05
        grasp_pose_4.pose.position.z = 0.6300246315623539
        grasp_pose_4.pose.orientation.x = -0.7068005682823383
        grasp_pose_4.pose.orientation.y = 0.7074128615379971
        grasp_pose_4.pose.orientation.z = -2.5258894636704497e-06
        grasp_pose_4.pose.orientation.w = -7.953924864631972e-08

        grasp_states = {
            (False, False): grasp_pose_1,
            (False, True): grasp_pose_2,
            (True, False): grasp_pose_3,
            (True, True): grasp_pose_4,
        }

        target_pose = PoseStamped()
        target_pose.pose.position.x = 1
        target_pose.pose.position.z = 0.7

        for from_above_mode in from_above_modes:
            for align_vertical_mode in align_vertical_modes:
                zero_pose.motion_goals.add_motion_goal(motion_goal_class='GraspObject',
                                                       goal_pose=target_pose,
                                                       from_above=from_above_mode,
                                                       align_vertical=align_vertical_mode,
                                                       root_link='map',
                                                       tip_link='hand_palm_link')

                zero_pose.allow_self_collision()
                zero_pose.plan_and_execute()
                m_P_g = (god_map.world.
                         compute_fk_pose('map', 'hand_gripper_tool_frame'))

                compare_poses(m_P_g.pose, grasp_states[from_above_mode, align_vertical_mode].pose)

    def test_vertical_motion_up(self, zero_pose: HSRTestWrapper):

        vertical_motion_pose = PoseStamped()
        vertical_motion_pose.header.frame_id = 'map'
        vertical_motion_pose.pose.position.x = 0.17102731790942596
        vertical_motion_pose.pose.position.y = -0.13231521471220506
        vertical_motion_pose.pose.position.z = 0.7119274770524749
        vertical_motion_pose.pose.orientation.x = 0.5067617681482114
        vertical_motion_pose.pose.orientation.y = -0.45782201564184877
        vertical_motion_pose.pose.orientation.z = 0.5271017946406412
        vertical_motion_pose.pose.orientation.w = 0.5057224638312487

        zero_pose.motion_goals.add_motion_goal(motion_goal_class='TakePose',
                                               pose_keyword='park')

        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

        sleep = zero_pose.monitors.add_sleep(seconds=0.1)
        local_min = zero_pose.monitors.add_local_minimum_reached(stay_true=False)

        action = ContextTypes.context_action.value(content=ContextActionModes.grasping.value)
        context = {'action': action}
        zero_pose.motion_goals.add_motion_goal(motion_goal_class='VerticalMotion',
                                               context=context,
                                               distance=0.02,
                                               root_link='base_footprint',
                                               tip_link='hand_palm_link',
                                               start_condition=sleep,
                                               end_condition=local_min)

        zero_pose.monitors.add_end_motion(start_condition=f'{sleep} and {local_min}')

        zero_pose.allow_self_collision()
        zero_pose.execute(add_local_minimum_reached=False)

        m_P_g = (god_map.world.
                 compute_fk_pose('map', 'hand_gripper_tool_frame'))

        compare_poses(m_P_g.pose, vertical_motion_pose.pose)

    def test_retracting_hand(self, zero_pose: HSRTestWrapper):

        retracting_hand_pose = PoseStamped()
        retracting_hand_pose.header.frame_id = 'map'
        retracting_hand_pose.pose.position.x = 0.14963260254170513
        retracting_hand_pose.pose.position.y = 0.16613649117825122
        retracting_hand_pose.pose.position.z = 0.6717532654948288
        retracting_hand_pose.pose.orientation.x = 0.5066648708788183
        retracting_hand_pose.pose.orientation.y = -0.45792002831875167
        retracting_hand_pose.pose.orientation.z = 0.5270228996549048
        retracting_hand_pose.pose.orientation.w = 0.5058130282241059

        zero_pose.motion_goals.add_motion_goal(motion_goal_class='TakePose',
                                               pose_keyword='park')

        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

        sleep = zero_pose.monitors.add_sleep(seconds=0.1)
        local_min = zero_pose.monitors.add_local_minimum_reached(stay_true=False)

        zero_pose.motion_goals.add_motion_goal(motion_goal_class='Retracting',
                                               distance=0.3,
                                               reference_frame='hand_palm_link',
                                               root_link='map',
                                               tip_link='hand_palm_link',
                                               start_condition=sleep)

        zero_pose.monitors.add_end_motion(start_condition=f'{local_min} and {sleep}')

        zero_pose.allow_self_collision()
        zero_pose.execute(add_local_minimum_reached=False)

        m_P_g = (god_map.world.
                 compute_fk_pose('map', 'hand_gripper_tool_frame'))

        compare_poses(m_P_g.pose, retracting_hand_pose.pose)

    def test_retracting_base(self, zero_pose: HSRTestWrapper):

        retraction_base_pose = PoseStamped()
        retraction_base_pose.header.frame_id = 'map'
        retraction_base_pose.pose.position.x = -0.12533144864637413
        retraction_base_pose.pose.position.y = 0.07795010184370622
        retraction_base_pose.pose.position.z = 0.894730930853242
        retraction_base_pose.pose.orientation.x = 0.014859073808224462
        retraction_base_pose.pose.orientation.y = -0.00015418547016511882
        retraction_base_pose.pose.orientation.z = 0.9998893945231346
        retraction_base_pose.pose.orientation.w = -0.0006187669689175172

        sleep = zero_pose.monitors.add_sleep(seconds=0.1)
        local_min = zero_pose.monitors.add_local_minimum_reached(stay_true=False)

        zero_pose.motion_goals.add_motion_goal(motion_goal_class='Retracting',
                                               distance=0.3,
                                               reference_frame='base_footprint',
                                               root_link='map',
                                               tip_link='hand_palm_link',
                                               start_condition=sleep)

        zero_pose.monitors.add_end_motion(start_condition=f'{local_min} and {sleep}')

        zero_pose.allow_self_collision()
        zero_pose.execute(add_local_minimum_reached=False)

        m_P_g = (god_map.world.
                 compute_fk_pose('map', 'hand_gripper_tool_frame'))

        compare_poses(m_P_g.pose, retraction_base_pose.pose)

    def test_align_height(self, zero_pose: HSRTestWrapper):
        execute_from_above = [False, True]

        align_pose1 = PoseStamped()
        align_pose1.header.frame_id = 'map'
        align_pose1.pose.position.x = 0.3670559556308583
        align_pose1.pose.position.y = 0.00022361096354857893
        align_pose1.pose.position.z = 0.7728331262049145
        align_pose1.pose.orientation.x = 0.6930355696535618
        align_pose1.pose.orientation.y = 0.0002441417024468236
        align_pose1.pose.orientation.z = 0.720903306535367
        align_pose1.pose.orientation.w = -0.0002494316878550612

        align_pose2 = PoseStamped()
        align_pose2.header.frame_id = 'map'
        align_pose2.pose.position.x = 0.2943309402390854
        align_pose2.pose.position.y = -0.0004960369085802845
        align_pose2.pose.position.z = 0.7499314955573722
        align_pose2.pose.orientation.x = 0.999999932400925
        align_pose2.pose.orientation.y = 0.0003514656228904682
        align_pose2.pose.orientation.z = -0.00010802805208618605
        align_pose2.pose.orientation.w = 3.7968463309867553e-08

        align_states = {
            False: align_pose1,
            True: align_pose2,
        }

        for mode in execute_from_above:
            zero_pose.motion_goals.add_motion_goal(motion_goal_class='TakePose',
                                                   pose_keyword='pre_align_height')

            zero_pose.allow_self_collision()
            zero_pose.plan_and_execute()

            action = ContextTypes.context_action.value(content=ContextActionModes.grasping.value)
            from_above = ContextTypes.context_from_above.value(content=mode)
            context = {'action': action, 'from_above': from_above}

            target_pose = PoseStamped()
            target_pose.header.frame_id = 'map'
            target_pose.pose.position.x = 1
            target_pose.pose.position.z = 0.7

            zero_pose.motion_goals.add_motion_goal(motion_goal_class='AlignHeight',
                                                   context=context,
                                                   object_name='',
                                                   goal_pose=target_pose,
                                                   object_height=0.1,
                                                   root_link='map',
                                                   tip_link='hand_palm_link')

            zero_pose.allow_self_collision()
            zero_pose.plan_and_execute()
            cord_data = (god_map.world.
                         compute_fk_pose('map', 'hand_gripper_tool_frame'))

            compare_poses(cord_data.pose, align_states[mode].pose)

    # FIXME: Tilting doesn't work. SuTuRo-Goal does not tilt.
    # Maybe change compare poses to finger tips and not tool_frame
    # def test_tilting(self, zero_pose: HSRTestWrapper):
    #     directions = ['left', 'right']
    #
    #     # Orientation for tilt_pose 1 needs to be negative despite given parameters being returned as positives...
    #     tilt_pose1 = PoseStamped()
    #     tilt_pose1.header.frame_id = 'map'
    #     tilt_pose1.pose.position.x = 0.3862282703183651
    #     tilt_pose1.pose.position.y = 0.07997985276116013
    #     tilt_pose1.pose.position.z = 0.7144424174771254
    #     tilt_pose1.pose.orientation.x = -0.4443082797691649
    #     tilt_pose1.pose.orientation.y = 0.5553960633877594
    #     tilt_pose1.pose.orientation.z = -0.5090628713771196
    #     tilt_pose1.pose.orientation.w = -0.4847477264384307
    #
    #     tilt_pose2 = PoseStamped()
    #     tilt_pose2.header.frame_id = 'map'
    #     tilt_pose2.pose.position.x = -0.04230309478138269
    #     tilt_pose2.pose.position.y = 0.07997985276116013
    #     tilt_pose2.pose.position.z = 0.8027971124538672
    #     tilt_pose2.pose.orientation.x = 0.36673529634187035
    #     tilt_pose2.pose.orientation.y = 0.6191556216911683
    #     tilt_pose2.pose.orientation.z = -0.5675033717108586
    #     tilt_pose2.pose.orientation.w = 0.4001143107189125
    #
    #     tilt_states = {
    #         'left': tilt_pose1,
    #         'right': tilt_pose2,
    #     }
    #
    #     for direction in directions:
    #         sleep = zero_pose.monitors.add_sleep(seconds=0.1)
    #         local_min = zero_pose.monitors.add_local_minimum_reached(stay_true=False)
    #
    #         zero_pose.motion_goals.add_motion_goal(motion_goal_class='Tilting',
    #                                                direction=direction,
    #                                                angle=1.4,
    #                                                start_condition='',
    #                                                end_condition=local_min)
    #
    #         zero_pose.monitors.add_end_motion(start_condition=f'{sleep} and {local_min}')
    #
    #         zero_pose.allow_self_collision()
    #         zero_pose.execute(add_local_minimum_reached=False)
    #
    #         cord_data = (god_map.world.
    #                      compute_fk_pose('map', 'hand_l_finger_tip_frame'))
    #
    #         compare_poses(cord_data.pose, tilt_states[direction].pose)

    def test_take_pose(self, zero_pose: HSRTestWrapper):
        poses = ['park', 'perceive', 'assistance', 'pre_align_height', 'carry']

        park_pose = PoseStamped()
        park_pose.header.frame_id = 'map'
        park_pose.pose.position.x = 0.1710261260244742
        park_pose.pose.position.y = -0.13231889092341187
        park_pose.pose.position.z = 0.6919283778314267
        park_pose.pose.orientation.x = 0.5067619888164565
        park_pose.pose.orientation.y = -0.45782179605285284
        park_pose.pose.orientation.z = 0.5271015813648557
        park_pose.pose.orientation.w = 0.5057226637915272

        perceive_pose = PoseStamped()
        perceive_pose.header.frame_id = 'map'
        perceive_pose.pose.position.x = 0.1710444625574895
        perceive_pose.pose.position.y = 0.2883150465871069
        perceive_pose.pose.position.z = 0.9371745637108605
        perceive_pose.pose.orientation.x = -0.5063851509844108
        perceive_pose.pose.orientation.y = -0.457448402898974
        perceive_pose.pose.orientation.z = -0.527458211023338
        perceive_pose.pose.orientation.w = 0.5060660758949697

        assistance_pose = PoseStamped()
        assistance_pose.header.frame_id = 'map'
        assistance_pose.pose.position.x = 0.18333071333185327
        assistance_pose.pose.position.y = -0.1306120975368269
        assistance_pose.pose.position.z = 0.7050680498627263
        assistance_pose.pose.orientation.x = 0.024667116882362873
        assistance_pose.pose.orientation.y = -0.6819662708507778
        assistance_pose.pose.orientation.z = 0.7305124281436971
        assistance_pose.pose.orientation.w = -0.025790135598626814

        pre_align_height_pose = PoseStamped()
        pre_align_height_pose.header.frame_id = 'map'
        pre_align_height_pose.pose.position.x = 0.36718508844870135
        pre_align_height_pose.pose.position.y = 0.07818733568602311
        pre_align_height_pose.pose.position.z = 0.6872325515876044
        pre_align_height_pose.pose.orientation.x = 0.6925625964573222
        pre_align_height_pose.pose.orientation.y = 0.0008342119786388634
        pre_align_height_pose.pose.orientation.z = 0.7213572801204168
        pre_align_height_pose.pose.orientation.w = 0.0001688074098573283

        carry_pose = PoseStamped()
        carry_pose.header.frame_id = 'map'
        carry_pose.pose.position.x = 0.4997932992635221
        carry_pose.pose.position.y = 0.06601541592028287
        carry_pose.pose.position.z = 0.6519470331487148
        carry_pose.pose.orientation.x = 0.49422863353080027
        carry_pose.pose.orientation.y = 0.5199402328561551
        carry_pose.pose.orientation.z = 0.4800020391690775
        carry_pose.pose.orientation.w = 0.5049735185624021

        assert_poses = {
            'park': park_pose.pose,
            'perceive': perceive_pose.pose,
            'assistance': assistance_pose.pose,
            'pre_align_height': pre_align_height_pose.pose,
            'carry': carry_pose.pose
        }

        for pose in poses:
            zero_pose.motion_goals.add_motion_goal(motion_goal_class='TakePose',
                                                   pose_keyword=pose)

            zero_pose.allow_self_collision()
            zero_pose.plan_and_execute()

            m_P_g = (god_map.world.
                     compute_fk_pose('map', 'hand_gripper_tool_frame'))

            compare_poses(m_P_g.pose, assert_poses[pose])

    # # TODO: If ever relevant for SuTuRo, add proper Test behaviour
    # def test_mixing(self, zero_pose: HSRTestWrapper):
    #     # FIXME: Cant use traj_time_in_seconds in standalone mode
    #     zero_pose.motion_goals.add_motion_goal(motion_goal_class='Mixing',
    #                                            mixing_time=20)
    #
    #     zero_pose.allow_self_collision()
    #     zero_pose.plan_and_execute()
    #
    # def test_joint_rotation_goal_continuous(self, zero_pose: HSRTestWrapper):
    #     # FIXME: Use compare_pose similar to other tests
    #     # FIXME: Cant use traj_time_in_seconds in standalone mode
    #     zero_pose.motion_goals.add_motion_goal(motion_goal_class='JointRotationGoalContinuous',
    #                                            joint_name='arm_roll_joint',
    #                                            joint_center=0.0,
    #                                            joint_range=0.2,
    #                                            trajectory_length=20,
    #                                            target_speed=1,
    #                                            period_length=1.0)
    #
    #     zero_pose.allow_self_collision()
    #     zero_pose.plan_and_execute()

    def test_keep_rotation_goal(self, zero_pose: HSRTestWrapper):

        keep_rotation_pose = PoseStamped()
        keep_rotation_pose.header.frame_id = 'map'
        keep_rotation_pose.pose.position.x = 0.9402845292991675
        keep_rotation_pose.pose.position.y = -0.7279803708852316
        keep_rotation_pose.pose.position.z = 0.8994121023446626
        keep_rotation_pose.pose.orientation.x = 0.015000397751939919
        keep_rotation_pose.pose.orientation.y = -2.1716350146486636e-07
        keep_rotation_pose.pose.orientation.z = 0.999887487627967
        keep_rotation_pose.pose.orientation.w = 1.2339723016403797e-05

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.position.y = -1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi / 2, [0, 0, 1]))
        zero_pose.set_cart_goal(base_goal, root_link=god_map.world.root_link_name, tip_link='base_footprint')

        zero_pose.motion_goals.add_motion_goal(motion_goal_class='KeepRotationGoal',
                                               tip_link='hand_palm_link')

        zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

        m_P_g = (god_map.world.
                 compute_fk_pose('map', 'hand_gripper_tool_frame'))

        compare_poses(m_P_g.pose, keep_rotation_pose.pose)
