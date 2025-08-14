from typing import Optional

from giskardpy import casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.goals.unlatch_door import UnlatchDoor
from giskardpy.motion_statechart.monitors.joint_monitors import JointGoalReached
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.task import WEIGHT_ABOVE_CA


class OpenDoorGoal(Goal):
    def __init__(self,
                 tip_link: PrefixName,
                 handle_name: PrefixName,
                 hinge_limit: float,
                 handle_limit: Optional[float] = None,
                 root_link: PrefixName = None,
                 tip_normal: cas.Vector3 = None,
                 goal_normal: cas.Vector3 = None,
                 name: str = None):
        """
        Use this, if you have grasped a door handle and want to open the door and handle

        :param tip_link: end effector that is grasping the handle
        :param handle_name: link that is grasped by the tip_link
        :param name: name of the goal
        :param handle_limit: Limit for how far the handle will be opened
        :param hinge_limit: Limit for how far the door-hinge will be opened
        :param root_link: Root-link for alignment of gripper to handle
        :param tip_normal: Gripper axis that is to be aligned with goal_normal
        :param goal_normal: Handle axis that is to be aligned with tip_normal
        """
        if name is None:
            name = 'OpenDoorGoal'
        if tip_normal is None:
            tip_normal = cas.Vector3()
            tip_normal.reference_frame = god_map.world.search_for_link_name('base_link', 'hsrb')
            tip_normal.x = 1
        if goal_normal is None:
            goal_normal = cas.Vector3()
            goal_normal.reference_frame = handle_name
            goal_normal.z = -1
        if root_link is None:
            root_link = god_map.world.search_for_link_name('map')
        super().__init__(name=name)

        handle_frame_id = god_map.world.get_movable_parent_joint(handle_name)
        link_id = god_map.world.get_parent_link_of_joint(handle_frame_id)
        door_hinge_id = god_map.world.get_movable_parent_joint(link_id)

        min_limit_hinge, max_limit_hinge = god_map.world.compute_joint_limits(door_hinge_id, 0)

        if hinge_limit is None:
            limit_hinge = min_limit_hinge
        else:
            limit_hinge = max(min_limit_hinge, hinge_limit)

        unlatch_door = UnlatchDoor(tip_link=tip_link,
                                   handle_name=handle_name,
                                   handle_limit=handle_limit)
        self.end_condition = unlatch_door
        self.add_goal(unlatch_door)

        jpl = JointPositionList(goal_state={door_hinge_id: max_limit_hinge},
                                weight=WEIGHT_ABOVE_CA,
                                name='DoorHinge')
        jpl.end_condition = unlatch_door
        self.add_task(jpl)

        apl = AlignPlanes(root_link=root_link,
                          tip_link=tip_normal.reference_frame,
                          goal_normal=goal_normal,
                          tip_normal=tip_normal,
                          name='AlignBaseWithDoor')
        apl.start_condition = unlatch_door
        self.add_task(apl)

        open_goal2 = Open(tip_link=tip_link,
                          environment_link=link_id,
                          goal_joint_state=limit_hinge,
                          name='OpenHinge',
                          max_velocity=0.5)
        open_goal2.start_condition = unlatch_door
        self.add_goal(open_goal2)

        goal_state = {door_hinge_id: limit_hinge}
        hinge_state_monitor = JointGoalReached(name='HingeMonitor', goal_state=goal_state)
        hinge_state_monitor.start_condition = unlatch_door
        self.add_monitor(hinge_state_monitor)

        self.observation_expression = hinge_state_monitor.observation_expression
