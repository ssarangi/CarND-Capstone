#!/usr/bin/env python

# Global packages
import math
import numpy as np
import rospy
import scipy.spatial
import time
import yaml

# Local project packages
import waypoint_lib.helper as helper
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped

from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from styx_msgs.msg import TrafficLightArray, TrafficLight


'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_VEL = 11.1


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', TrafficLight, self.upcoming_traffic_light_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb, queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.vehicle_dbw_enabled_cb)


        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.stop_line_waypoint_pub = rospy.Publisher('stop_line_waypoint', Int32, queue_size=1)

        # TODO: Add other member variables you need below
        # A list of all the waypoints of the track as reported by master node.
        self.waypoints = None
        self.current_velocity = 0.0

        self.waypoints_kdtree = None
        self.current_traffic_light = None

        self.current_pose = None
        self.stop_line_positions = None
        self.stop_line_positions_kdtree = None

        self.initialize_stop_line_positions_kdtree()
        self.current_velocity = 0.0
        self.deceleration_started = False
        self.deceleration_waypoints = None
        self.dbw_enabled = False
        rospy.spin()

    def vehicle_dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg

    def initialize_stop_line_positions_kdtree(self):
        self.stop_line_positions = self.config['stop_line_positions']
        self.stop_line_positions_kdtree = scipy.spatial.cKDTree(self.stop_line_positions)

    def current_velocity_cb(self, curr_vel_msg):
        self.current_velocity = curr_vel_msg.twist.linear.x

    def set_velocities(self, waypoints, velocities, start_wp_idx, end_wp_idx):
        vel_idx = 0
        for wp_idx in range(start_wp_idx, end_wp_idx + 1):
            self.set_waypoint_velocity(waypoints, wp_idx, velocities[vel_idx])
            vel_idx += 1

    def set_linear_distribution_velocities(self, waypoints, start_vel, end_velocity, start_wp_idx, end_wp_idx):
        # Compute the total distances leading up to the stop point
        # distances = [self.distance(waypoints, i, i+1)
        #              for i in range(start_wp_idx, end_wp_idx)]

        # total_distance = int(math.floor(sum(distances)))
	if end_wp_idx > start_wp_idx - 1: delta_idx = end_wp_idx - start_wp_idx + 1
	else: delta_idx = end_wp_idx + len(self.waypoints) - start_wp_idx + 1
        velocities = np.linspace(start_vel, end_velocity, delta_idx)
        self.set_velocities(waypoints, velocities, start_wp_idx, end_wp_idx)

    def set_smooth_acceleration_to_speed_limit(self, start_wp_idx, stop_wp_idx):
        new_waypoints = self.waypoints[:]  # Copy the list
        self.set_linear_distribution_velocities(new_waypoints,
                                                min(MAX_VEL, self.current_velocity + 1.0),  # Increase the current velocity
                                                MAX_VEL,
                                                start_wp_idx + 1,
                                                stop_wp_idx)

        return new_waypoints

    def set_velocity_leading_to_stop_point(self, start_wp_idx, stop_point_idx):
        new_waypoints = self.waypoints[:]  # Copy the list
        self.set_linear_distribution_velocities(new_waypoints,
                                                self.current_velocity,
                                                0.0,
                                                start_wp_idx + 1,
                                                stop_point_idx)

        return new_waypoints

    def behavior_lights_green(self, closest_wp_idx):
        # If the lights are green just continue on the same path.
        if self.deceleration_started:
            self.deceleration_started = False
            self.deceleration_waypoints = None

        new_waypoints = self.set_smooth_acceleration_to_speed_limit(closest_wp_idx,
                                                                    closest_wp_idx + LOOKAHEAD_WPS)

        return new_waypoints

    def pose_cb(self, pose):
        if self.waypoints is None:
            rospy.error('No base_waypoints have been received by master')
            return

        if not self.dbw_enabled:
            return

        self.current_pose = pose

        # Compute the index of the waypoint closest to the current pose.
        closest_wp_idx = helper.next_waypoint_index_kdtree(self.current_pose.pose, self.waypoints_kdtree)

        # Find the closest stop line position if we have to stop the car.
        _, stop_line_idx = self.stop_line_positions_kdtree.query([pose.pose.position.x, pose.pose.position.y])
        stop_line_positions = self.stop_line_positions[stop_line_idx]

        # Find the closest waypoint index for the stop line position
        stop_line_pose = PoseStamped()
        stop_line_pose.pose.position.x = stop_line_positions[0]
        stop_line_pose.pose.position.y = stop_line_positions[1]
        stop_line_pose.pose.orientation = pose.pose.orientation
        stop_line_waypoint_idx = helper.next_waypoint_index_kdtree(stop_line_pose.pose, self.waypoints_kdtree)

        self.stop_line_waypoint_pub.publish(Int32(stop_line_waypoint_idx))

        if closest_wp_idx == stop_line_waypoint_idx or self.current_velocity == 0.0:
            rospy.logerr('Stop Line waypoint reached. Deceleration Sequence Stopped. Current Vel: %s', self.current_velocity)
            self.deceleration_started = False
            self.deceleration_waypoints = None

        # If the light is RED or YELLOW then slowly decrease the speed.
        if self.current_traffic_light is not None:
            rospy.logwarn('Light color: %s', helper.get_traffic_light_color(self.current_traffic_light.state))
            if self.current_traffic_light.state == 0 or self.current_traffic_light.state == 1:
                if helper.deceleration_rate(self.current_velocity, self.distance(self.waypoints,closest_wp_idx,stop_line_waypoint_idx)) > 0.1:
                    if not self.deceleration_started:
                        rospy.logwarn('Deceleration Sequence Started')
                        new_waypoints = self.set_velocity_leading_to_stop_point(closest_wp_idx, stop_line_waypoint_idx)
                        self.deceleration_started = True
                        self.deceleration_waypoints = new_waypoints
                    else:
                        rospy.logwarn('Using DECELERATION WAYPOINTS CALCULATED BEFORE')
                        new_waypoints = self.deceleration_waypoints
                elif closest_wp_idx != stop_line_waypoint_idx:
                    # Simulate the behavior of a green light
                    new_waypoints = self.behavior_lights_green(closest_wp_idx)
                else:
                    rospy.logerr('NOT PUBLISHING ANYTHING SINCE ASSUMING CAR IS AT STOP')
                    return  # Do not publish anything. The car is at a STOP and we don't need to do anything
            else:
                # Lights are green
                new_waypoints = self.behavior_lights_green(closest_wp_idx)
        else:
            new_waypoints = self.behavior_lights_green(closest_wp_idx)

        if stop_line_waypoint_idx - closest_wp_idx < 5:
            rospy.logerr('Current Waypoint: %s Current Velocity: %s', closest_wp_idx, self.current_velocity)
            for i in range(stop_line_waypoint_idx - 5, stop_line_waypoint_idx + 1):
                rospy.loginfo('%s, %s', i, new_waypoints[i].twist.twist.linear.x)

        # Find number of waypoints ahead dictated by LOOKAHEAD_WPS. However, if the car is stopped then don't accelerate
        next_wps = new_waypoints[closest_wp_idx + 1:closest_wp_idx + LOOKAHEAD_WPS]
        self.publish(next_wps)

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        # Create a numpy version of the waypoints
        np_waypoints = helper.create_numpy_repr(self.waypoints)
        self.waypoints_kdtree = scipy.spatial.cKDTree(np_waypoints, leafsize=5)
        rospy.loginfo('Created KDTree for vehicle waypoints for fast NN query')

    def upcoming_traffic_light_cb(self, msg):
        self.current_traffic_light = msg

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def publish(self, waypoints):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(time.time())
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
