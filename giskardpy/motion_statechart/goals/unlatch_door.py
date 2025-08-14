from typing import Optional

from giskardpy.data_types.data_types import PrefixName
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.monitors.joint_monitors import JointGoalReached
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList


class UnlatchDoor(Goal):
    def __init__(self,
                 tip_link: PrefixName,
                 handle_name: PrefixName,
                 name: Optional[str] = None,
                 handle_limit: Optional[float] = None):
        if name is None:
            name = UnlatchDoor.__name__
        super().__init__(name=name)

        handle_name = handle_name
        handle_frame_id = god_map.world.get_movable_parent_joint(handle_name)
        _, max_limit_handle = god_map.world.compute_joint_limits(handle_frame_id, 0)

        if handle_limit is None:
            limit_handle = max_limit_handle
        else:
            limit_handle = min(max_limit_handle, handle_limit)

        handle_state = {handle_frame_id: limit_handle}
        handle_state_monitor = JointGoalReached(goal_state=handle_state,
                                                threshold=0.005,
                                                name=f'{name}_handle_joint_monitor')
        self.add_monitor(handle_state_monitor)

        open_goal = Open(tip_link=tip_link,
                         environment_link=handle_name,
                         goal_joint_state=limit_handle,
                         name='OpenHandle',
                         max_velocity=0.3)
        self.add_goal(open_goal)

        self.observation_expression = handle_state_monitor.observation_expression
