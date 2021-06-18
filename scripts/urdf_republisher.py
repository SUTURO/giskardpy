#!/usr/bin/env python
import rospy

rospy.set_param('/giskard/robot_description', rospy.get_param('/robot_description'))