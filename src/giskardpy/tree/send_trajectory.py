import control_msgs
from rospy import ROSException
from rostopic import ROSTopicException
from sensor_msgs.msg import JointState

try:
    import pr2_controllers_msgs.msg
except ImportError:
    pass
import py_trees
import rospy
import rostopic
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, FollowJointTrajectoryResult, \
    JointTrajectoryControllerState
from py_trees_ros.actions import ActionClient

import giskardpy.identifier as identifier
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.logging import loginfo


class SendFollowJointTrajectory(ActionClient, GiskardBehavior):
    error_code_to_str = {value: name for name, value in vars(FollowJointTrajectoryResult).items() if
                         isinstance(value, int)}

    supported_action_types = [control_msgs.msg.FollowJointTrajectoryAction,
                              pr2_controllers_msgs.msg.JointTrajectoryAction]
    supported_state_types = [control_msgs.msg.JointTrajectoryControllerState,
                             pr2_controllers_msgs.msg.JointTrajectoryControllerState]

    def __init__(self, name, namespace, state_topic, fill_velocity_values=True):
        GiskardBehavior.__init__(self, name)
        self.action_namespace = namespace
        self.fill_velocity_values = fill_velocity_values

        loginfo('Waiting for action server \'{}\' to appear.'.format(self.action_namespace))
        action_msg_type = None
        while not action_msg_type:
            try:
                action_msg_type, _, _ = rostopic.get_topic_class('{}/goal'.format(self.action_namespace))
                try:
                    action_msg_type = eval(action_msg_type._type.replace('/', '.msg.')[:-4])
                    if action_msg_type not in self.supported_action_types:
                        raise TypeError()
                except Exception as e:
                    raise TypeError('Action server of type \'{}\' is not supported. '
                                    'Must be one of: {}'.format(action_msg_type, self.supported_action_types))
            except ROSTopicException as e:
                logging.logwarn('Couldn\'t connect to {}. Is it running?'.format(self.action_namespace))
                rospy.sleep(2)

        ActionClient.__init__(self, name, action_msg_type, None, self.action_namespace)
        loginfo('Successfully connected to \'{}\'.'.format(self.action_namespace))

        loginfo('Waiting for state topic \'{}\' to appear.'.format(state_topic))
        msg = None
        while not msg:
            try:
                status_msg_type, _, _ = rostopic.get_topic_class(state_topic)
                if status_msg_type not in self.supported_state_types:
                    raise TypeError('State topic of type \'{}\' is not supported. '
                                    'Must be one of: {}'.format(status_msg_type, self.supported_state_types))
                msg = rospy.wait_for_message(state_topic, status_msg_type, timeout=2.0)
                if isinstance(msg, JointState):
                    self.controlled_joints = msg.name
                elif isinstance(msg, control_msgs.msg.JointTrajectoryControllerState) \
                    or isinstance(msg, pr2_controllers_msgs.msg.JointTrajectoryControllerState):
                    self.controlled_joints = msg.joint_names
            except ROSException as e:
                logging.logwarn('Couldn\'t connect to {}. Is it running?'.format(state_topic))
        self.world.register_controlled_joints(self.controlled_joints)
        loginfo('Received controlled joints from \'{}\'.'.format(state_topic))


    def initialise(self):
        super(SendFollowJointTrajectory, self).initialise()
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        goal = FollowJointTrajectoryGoal()
        sample_period = self.get_god_map().get_data(identifier.sample_period)
        goal.trajectory = trajectory.to_msg(sample_period, self.controlled_joints, self.fill_velocity_values)
        self.action_goal = goal

    def update(self):
        """
        Check only to see whether the underlying action server has
        succeeded, is running, or has cancelled/aborted for some reason and
        map these to the usual behaviour return states.

        overriding this shit because of the fucking prints
        """
        self.logger.debug("{0}.update()".format(self.__class__.__name__))
        if not self.action_client:
            self.feedback_message = "no action client, did you call setup() on your tree?"
            return py_trees.Status.INVALID
        # pity there is no 'is_connected' api like there is for c++
        if not self.sent_goal:
            self.action_client.send_goal(self.action_goal)
            logging.loginfo('Sending trajectory to \'{}\'.'.format(self.action_namespace))
            self.sent_goal = True
            self.feedback_message = "sent goal to the action server"
            return py_trees.Status.RUNNING
        if self.action_client.get_state() == GoalStatus.ABORTED:
            result = self.action_client.get_result()
            self.feedback_message = self.error_code_to_str[result.error_code]
            logging.logerr('fail {}'.format(self.feedback_message))
            return py_trees.Status.FAILURE
        result = self.action_client.get_result()
        if result:
            self.feedback_message = "goal reached"
            logging.loginfo('{} successfully executed the trajectory.'.format(self.action_namespace))
            return py_trees.Status.SUCCESS
        else:
            self.feedback_message = "moving"
            return py_trees.Status.RUNNING
