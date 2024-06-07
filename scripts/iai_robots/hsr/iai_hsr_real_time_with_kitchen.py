#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.hsr import WorldWithHSRConfig, HSRCollisionAvoidanceConfig, \
    HSRVelocityInterface, SuturoArenaWithHSRConfig

if __name__ == '__main__':
    rospy.init_node('giskard')
    debug_mode = rospy.get_param('~debug_mode', False)
    giskard = Giskard(world_config=SuturoArenaWithHSRConfig(),
                      collision_avoidance_config=HSRCollisionAvoidanceConfig(),
                      robot_interface_config=HSRVelocityInterface(),
                      behavior_tree_config=ClosedLoopBTConfig(publish_free_variables=True, debug_mode=debug_mode,
                                                              add_tf_pub=True))
    giskard.live()