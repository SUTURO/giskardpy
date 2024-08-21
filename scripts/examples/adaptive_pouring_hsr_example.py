import rospy
from giskardpy.python_interface.python_interface import GiskardWrapper
from geometry_msgs.msg import PoseStamped, Vector3Stamped, Quaternion, Point
from tf.transformations import quaternion_from_matrix

# One can run this directly after 'roslaunch giskardpy giskardpy_hsr_standalone.launch'
rospy.init_node('giskard_example')
giskard = GiskardWrapper()

start_pose = PoseStamped()
start_pose.header.frame_id = 'map'
start_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                 [0, -1, 0, 0],
                                                                 [1, 0, 0, 0],
                                                                 [0, 0, 0, 1]]))
start_pose.pose.position = Point(1, 0, 0.8)

tilt_axis = Vector3Stamped()
tilt_axis.header.frame_id = 'hand_palm_link'
tilt_axis.vector.z = 1

tilt_angle = 0.8

# Here the adaptive pouring is run without feedack as this would require a simulation setup with an interpreter
# to infer the correct feedback for the goal
# This will now move the tip_link to the start pose and then tilt it around tilt axis for tilt_angle rad.
# Using the similar method from the old python interface works the same.
giskard.motion_goals.add_adaptive_pouring(tip_link='hand_palm_link',
                                          root_link='map',
                                          start_pose=start_pose,
                                          tilt_axis=tilt_axis,
                                          tilt_angle=tilt_angle,
                                          with_feedback=False)
giskard.monitors.add_max_trajectory_length(30)
giskard.add_default_end_motion_conditions()
giskard.execute()
