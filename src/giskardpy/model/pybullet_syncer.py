from collections import defaultdict
from itertools import combinations
from time import time

import numpy as np
from giskard_msgs.msg import CollisionEntry
from tf.transformations import quaternion_matrix, quaternion_from_matrix

import giskardpy.model.pybullet_wrapper as pbw
from giskardpy import RobotName
from giskardpy.data_types import BiDict
from giskardpy.exceptions import PhysicsWorldException, UnknownBodyException
from giskardpy.model.world import SubWorldTree
from giskardpy.model.world import WorldTree
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import np_to_pose
from giskardpy.utils.utils import resolve_ros_iris


class PyBulletSyncer(object):
    def __init__(self, world, gui=False):
        pbw.start_pybullet(gui)
        self.object_name_to_bullet_id = BiDict()
        self.world = world # type: WorldTree
        self.collision_matrices = defaultdict(set)

    @property
    def robot(self):
        """
        :rtype: SubWorldTree
        """
        return self.world.groups[RobotName]

    @property
    def god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.world.god_map

    @profile
    def add_object(self, link):
        """
        :type link: giskardpy.model.world.Link
        """
        pose = self.fks[link.name]
        position = pose[:3]
        orientation = pose[4:]
        self.object_name_to_bullet_id[link.name] = pbw.load_urdf_string_into_bullet(link.as_urdf(),
                                                                                    position=position,
                                                                                    orientation=orientation)

    @profile
    def update_pose(self, link):
        pose = self.fks[link.name]
        position = pose[:3]
        orientation = pose[4:]
        pbw.resetBasePositionAndOrientation(self.object_name_to_bullet_id[link.name], position, orientation)

    def calc_collision_matrix(self, group_name, link_combinations=None, d=0.05, d2=0.0, num_rnd_tries=200):
        """
        :type group_name: str
        :param link_combinations: set with link name tuples
        :type link_combinations: set
        :param d: distance threshold to detect links that are always in collision
        :type d: float
        :param d2: distance threshold to find links that are sometimes in collision
        :type d2: float
        :param num_rnd_tries:
        :type num_rnd_tries: int
        :return: set of link name tuples which are sometimes in collision.
        :rtype: set
        """
        group = self.world.groups[group_name] # type: SubWorldTree
        if link_combinations is None:
            link_combinations = set(combinations(group.link_names_with_collisions, 2))
        # TODO computational expansive because of too many collision checks
        logging.loginfo(u'calculating self collision matrix')
        joint_state_tmp = group.state
        t = time()
        np.random.seed(1337)
        always = set()

        # find meaningless self-collisions
        for link_a, link_b in link_combinations:
            if group.are_linked(link_a, link_b) or \
                    link_a == link_b:
                    # link_a in self.ignored_pairs or \
                    # link_b in self.ignored_pairs or \
                    # (link_a, link_b) in self.ignored_pairs or \
                    # (link_b, link_a) in self.ignored_pairs:
                always.add((link_a, link_b))
        rest = link_combinations.difference(always)
        group.set_joint_state_to_zero()
        always = always.union(self.check_collisions(rest, d))
        rest = rest.difference(always)

        # find meaningful self-collisions
        group.set_min_joint_state()
        sometimes = self.check_collisions(rest, d2)
        rest = rest.difference(sometimes)
        group.set_max_joint_state()
        sometimes2 = self.check_collisions(rest, d2)
        rest = rest.difference(sometimes2)
        sometimes = sometimes.union(sometimes2)
        for i in range(num_rnd_tries):
            group.set_rnd_joint_state()
            sometimes2 = self.check_collisions(rest, d2)
            if len(sometimes2) > 0:
                rest = rest.difference(sometimes2)
                sometimes = sometimes.union(sometimes2)
        # sometimes = sometimes.union(self.added_pairs)
        logging.loginfo(u'calculated self collision matrix in {:.3f}s'.format(time() - t))
        group.state = joint_state_tmp

        self.collision_matrices[group_name] = sometimes
        return self.collision_matrices[group_name]

    def init_collision_matrix(self, group_name):
        self.sync()
        added_links = set(combinations(self.world.groups[group_name].link_names_with_collisions, 2))
        self.update_collision_matrix(group_name=group_name,
                                     added_links=added_links)

    def update_collision_matrix(self, group_name, added_links=None, removed_links=None):
        # if not self.load_self_collision_matrix(self.path_to_data_folder):
        if added_links is None:
            added_links = set()
        if removed_links is None:
            removed_links = set()
        # collision_matrix = {x for x in self.collision_matrices[group_name] if x[0] not in removed_links and
        #                                x[1] not in removed_links}
        collision_matrix = self.calc_collision_matrix(group_name, added_links)
        self.collision_matrices[group_name] = collision_matrix
        # self.safe_self_collision_matrix(self.path_to_data_folder)

    def check_collisions(self, link_combinations, distance):
        in_collision = set()
        self.sync_state()
        for link_a, link_b in link_combinations:
            if self.in_collision(link_a, link_b, distance):
                in_collision.add((link_a, link_b))
        return in_collision

    def in_collision(self, link_a, link_b, distance):
        link_id_a = self.object_name_to_bullet_id[link_a]
        link_id_b = self.object_name_to_bullet_id[link_b]
        return len(pbw.getClosestPoints(link_id_a, link_id_b, distance)) > 0

    @profile
    def sync_state(self):
        """
        :type world: giskardpy.model.world.WorldTree
        """
        pbw.deactivate_rendering()
        self.fks = self.world.compute_all_fks()
        for link_name, link in self.world.links.items():
            if link.has_collisions():
                self.update_pose(link)
        pbw.activate_rendering()

    def sync(self):
        """
        :type world: giskardpy.model.world.WorldTree
        """
        pbw.deactivate_rendering()
        self.object_name_to_bullet_id = BiDict()
        self.world.soft_reset()
        pbw.clear_pybullet()
        self.fks = self.world.compute_all_fks()
        for link_name, link in self.world.links.items():
            if link.has_collisions():
                self.add_object(link)
        pbw.activate_rendering()

    def are_entries_known(self, collision_goals):
        robot_name = RobotName
        robot_links = set(self.robot.link_names)
        for collision_entry in collision_goals:
            if collision_entry.body_b not in self.world.groups and not self.all_body_bs(collision_entry):
                raise UnknownBodyException(u'body b \'{}\' unknown'.format(collision_entry.body_b))
            if not self.all_robot_links(collision_entry):
                for robot_link in collision_entry.robot_links:
                    if robot_link not in robot_links:
                        raise UnknownBodyException(u'robot link \'{}\' unknown'.format(robot_link))
            if collision_entry.body_b == robot_name:
                for robot_link in collision_entry.link_bs:
                    if robot_link != CollisionEntry.ALL and robot_link not in robot_links:
                        raise UnknownBodyException(
                            u'link b \'{}\' of body \'{}\' unknown'.format(robot_link, collision_entry.body_b))
            elif not self.all_body_bs(collision_entry) and not self.all_link_bs(collision_entry):
                object_links = self.world.groups[collision_entry.body_b].link_names
                for link_b in collision_entry.link_bs:
                    if link_b not in object_links:
                        raise UnknownBodyException(
                            u'link b \'{}\' of body \'{}\' unknown'.format(link_b, collision_entry.body_b))

    def collision_goals_to_collision_matrix(self, collision_goals, min_dist):
        """
        :param collision_goals: list of CollisionEntry
        :type collision_goals: list
        :return: dict mapping (robot_link, body_b, link_b) -> min allowed distance
        :rtype: dict
        """
        collision_goals = self.verify_collision_entries(collision_goals)
        min_allowed_distance = {}
        for collision_entry in collision_goals:  # type: CollisionEntry
            if self.is_avoid_all_self_collision(collision_entry):
                min_allowed_distance.update(self.get_robot_collision_matrix(min_dist))
                continue
            assert len(collision_entry.robot_links) == 1
            assert len(collision_entry.link_bs) == 1
            key = (collision_entry.robot_links[0], collision_entry.body_b, collision_entry.link_bs[0])
            r_key = (collision_entry.link_bs[0], collision_entry.body_b, collision_entry.robot_links[0])
            if self.is_allow_collision(collision_entry):
                if self.all_link_bs(collision_entry):
                    for key2 in list(min_allowed_distance.keys()):
                        if key[0] == key2[0] and key[1] == key2[1]:
                            del min_allowed_distance[key2]
                elif key in min_allowed_distance:
                    del min_allowed_distance[key]
                elif r_key in min_allowed_distance:
                    del min_allowed_distance[r_key]

            elif self.is_avoid_collision(collision_entry):
                min_allowed_distance[key] = min_dist[key[0]]
            else:
                raise Exception('todo')
        return min_allowed_distance

    def get_robot_collision_matrix(self, min_dist):
        robot_name = self.robot.name
        collision_matrix = self.collision_matrices[RobotName]
        collision_matrix2 = {}
        for link1, link2 in collision_matrix:
            # FIXME should I use the minimum of both distances?
            if self.robot.link_order(link1, link2):
                collision_matrix2[link1, robot_name, link2] = min_dist[link1]
            else:
                collision_matrix2[link2, robot_name, link1] = min_dist[link1]
        return collision_matrix2

    def verify_collision_entries(self, collision_goals):
        for ce in collision_goals:  # type: CollisionEntry
            if ce.type in [CollisionEntry.ALLOW_ALL_COLLISIONS,
                           CollisionEntry.AVOID_ALL_COLLISIONS]:
                # logging.logwarn(u'ALLOW_ALL_COLLISIONS and AVOID_ALL_COLLISIONS deprecated, use AVOID_COLLISIONS and'
                #               u'ALLOW_COLLISIONS instead with ALL constant instead.')
                if ce.type == CollisionEntry.ALLOW_ALL_COLLISIONS:
                    ce.type = CollisionEntry.ALLOW_COLLISION
                else:
                    ce.type = CollisionEntry.AVOID_COLLISION

        for ce in collision_goals:  # type: CollisionEntry
            if CollisionEntry.ALL in ce.robot_links and len(ce.robot_links) != 1:
                raise PhysicsWorldException(u'ALL used in robot_links, but it\'s not the only entry')
            if CollisionEntry.ALL in ce.link_bs and len(ce.link_bs) != 1:
                raise PhysicsWorldException(u'ALL used in link_bs, but it\'s not the only entry')
            if ce.body_b == CollisionEntry.ALL and not self.all_link_bs(ce):
                raise PhysicsWorldException(u'if body_b == ALL, link_bs has to be ALL as well')

        self.are_entries_known(collision_goals)

        for ce in collision_goals:
            if not ce.robot_links:
                ce.robot_links = [CollisionEntry.ALL]
            if not ce.link_bs:
                ce.link_bs = [CollisionEntry.ALL]

        for i, ce in enumerate(reversed(collision_goals)):
            if self.is_avoid_all_collision(ce):
                collision_goals = collision_goals[len(collision_goals) - i - 1:]
                break
            if self.is_allow_all_collision(ce):
                collision_goals = collision_goals[len(collision_goals) - i:]
                break
        else:
            ce = CollisionEntry()
            ce.type = CollisionEntry.AVOID_COLLISION
            ce.robot_links = [CollisionEntry.ALL]
            ce.body_b = CollisionEntry.ALL
            ce.link_bs = [CollisionEntry.ALL]
            ce.min_dist = -1
            collision_goals.insert(0, ce)

        # split body bs
        collision_goals = self.split_body_b(collision_goals)

        # split robot links
        collision_goals = self.robot_related_stuff(collision_goals)

        # split link_bs
        collision_goals = self.split_link_bs(collision_goals)

        return collision_goals

    def split_link_bs(self, collision_goals):
        # FIXME remove the side effects of these three methods
        i = 0
        while i < len(collision_goals):
            collision_entry = collision_goals[i]
            if self.is_avoid_all_self_collision(collision_entry):
                i += 1
                continue
            if self.all_link_bs(collision_entry):
                if collision_entry.body_b == RobotName:
                    new_ces = []
                    link_bs = self.get_possible_collisions(list(collision_entry.robot_links)[0])
                elif [x for x in collision_goals[i:] if
                      x.robot_links == collision_entry.robot_links and
                      x.body_b == collision_entry.body_b and not self.all_link_bs(x)]:
                    new_ces = []
                    link_bs = self.world.groups[collision_entry.body_b].link_names_with_collisions
                else:
                    i += 1
                    continue
                collision_goals.remove(collision_entry)
                for link_b in link_bs:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = collision_entry.robot_links
                    ce.body_b = collision_entry.body_b
                    ce.link_bs = [link_b]
                    ce.min_dist = collision_entry.min_dist
                    new_ces.append(ce)
                for new_ce in new_ces:
                    collision_goals.insert(i, new_ce)
                i += len(new_ces)
                continue
            elif len(collision_entry.link_bs) > 1:
                collision_goals.remove(collision_entry)
                for link_b in collision_entry.link_bs:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = collision_entry.robot_links
                    ce.body_b = collision_entry.body_b
                    ce.link_bs = [link_b]
                    ce.min_dist = collision_entry.min_dist
                    collision_goals.insert(i, ce)
                i += len(collision_entry.link_bs)
                continue
            i += 1
        return collision_goals

    def get_possible_collisions(self, link):
        # TODO speed up by saving this
        possible_collisions = set()
        for link1, link2 in self.collision_matrices[RobotName]:
            if link == link1:
                possible_collisions.add(link2)
            elif link == link2:
                possible_collisions.add(link1)
        return possible_collisions

    def robot_related_stuff(self, collision_goals):
        i = 0
        # TODO why did i use controlled links?
        # controlled_robot_links = self.robot.get_controlled_links()
        controlled_robot_links = self.robot.link_names_with_collisions
        while i < len(collision_goals):
            collision_entry = collision_goals[i]
            if self.is_avoid_all_self_collision(collision_entry):
                i += 1
                continue
            if self.all_robot_links(collision_entry):
                collision_goals.remove(collision_entry)

                new_ces = []
                for robot_link in controlled_robot_links:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = [robot_link]
                    ce.body_b = collision_entry.body_b
                    ce.min_dist = collision_entry.min_dist
                    ce.link_bs = collision_entry.link_bs
                    new_ces.append(ce)

                for new_ce in new_ces:
                    collision_goals.insert(i, new_ce)
                i += len(new_ces)
                continue
            elif len(collision_entry.robot_links) > 1:
                collision_goals.remove(collision_entry)
                for robot_link in collision_entry.robot_links:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = [robot_link]
                    ce.body_b = collision_entry.body_b
                    ce.min_dist = collision_entry.min_dist
                    ce.link_bs = collision_entry.link_bs
                    collision_goals.insert(i, ce)
                i += len(collision_entry.robot_links)
                continue
            i += 1
        return collision_goals

    def split_body_b(self, collision_goals):
        i = 0
        while i < len(collision_goals):
            collision_entry = collision_goals[i]
            if self.all_body_bs(collision_entry):
                collision_goals.remove(collision_entry)
                new_ces = []
                for body_b in self.world.groups:
                    ce = CollisionEntry()
                    ce.type = collision_entry.type
                    ce.robot_links = collision_entry.robot_links
                    ce.min_dist = collision_entry.min_dist
                    ce.body_b = body_b
                    ce.link_bs = collision_entry.link_bs
                    new_ces.append(ce)
                for new_ce in reversed(new_ces):
                    collision_goals.insert(i, new_ce)
                i += len(new_ces)
                continue
            i += 1
        return collision_goals

    def all_robot_links(self, collision_entry):
        return CollisionEntry.ALL in collision_entry.robot_links and len(collision_entry.robot_links) == 1

    def all_link_bs(self, collision_entry):
        return CollisionEntry.ALL in collision_entry.link_bs and len(collision_entry.link_bs) == 1 or \
               not collision_entry.link_bs

    def all_body_bs(self, collision_entry):
        return collision_entry.body_b == CollisionEntry.ALL

    def is_avoid_collision(self, collision_entry):
        return collision_entry.type in [CollisionEntry.AVOID_COLLISION, CollisionEntry.AVOID_ALL_COLLISIONS]

    def is_allow_collision(self, collision_entry):
        return collision_entry.type in [CollisionEntry.ALLOW_COLLISION, CollisionEntry.ALLOW_ALL_COLLISIONS]

    def is_avoid_all_self_collision(self, collision_entry):
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_avoid_collision(collision_entry) \
               and self.all_robot_links(collision_entry) \
               and collision_entry.body_b == RobotName \
               and self.all_link_bs(collision_entry)

    def is_allow_all_self_collision(self, collision_entry):
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_allow_collision(collision_entry) \
               and self.all_robot_links(collision_entry) \
               and collision_entry.body_b == RobotName \
               and self.all_link_bs(collision_entry)

    def is_avoid_all_collision(self, collision_entry):
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_avoid_collision(collision_entry) \
               and self.all_robot_links(collision_entry) \
               and self.all_body_bs(collision_entry) \
               and self.all_link_bs(collision_entry)

    def is_allow_all_collision(self, collision_entry):
        """
        :type collision_entry: CollisionEntry
        :return: bool
        """
        return self.is_allow_collision(collision_entry) \
               and self.all_robot_links(collision_entry) \
               and self.all_body_bs(collision_entry) \
               and self.all_link_bs(collision_entry)


    # def __add_ground_plane(self):
    #     """
    #     Adds a ground plane to the Bullet World.
    #     """
    #     path = resolve_ros_iris(u'package://giskardpy/urdfs/ground_plane.urdf')
    #     plane = WorldObject.from_urdf_file(path)
    #     plane.set_name(self.ground_plane_name)
    #     self.add_object(plane)
