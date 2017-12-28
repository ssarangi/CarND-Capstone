#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from waypoint_lib import helper

import scipy.spatial
import tf
import time

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        self.bridge = CvBridge()
        self.is_carla = rospy.get_param("/is_carla", False)
        self.light_classifier = TLClassifier(self.is_carla)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.traffic_lights_kdtree = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        self.traffic_lights_sub = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_traffic_light_pub = rospy.Publisher('/traffic_waypoint', TrafficLight, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights
        np_light_waypoints = helper.create_numpy_repr(msg.lights)
        self.traffic_lights_kdtree = scipy.spatial.cKDTree(np_light_waypoints)
        rospy.loginfo('Created KDTree for light waypoints for fast NN query')
        self.traffic_lights_sub.unregister()
        self.traffic_lights_sub = None

    def publish_upcoming_light_state(self, idx, light_state):
        traffic_light = TrafficLight()
        traffic_light.header.frame_id = '/world'
        traffic_light.header.stamp = rospy.Time(time.time())
        traffic_light.idx = idx
        traffic_light.pose = self.lights[idx].pose
        traffic_light.state = light_state
        self.upcoming_traffic_light_pub.publish(traffic_light)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights(self.camera_image)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.publish_upcoming_light_state(light_wp, self.state)
        else:
            self.publish_upcoming_light_state(self.last_wp, self.last_state)
        self.state_count += 1

    def get_closest_traffic_light(self, pose):
        """Identifies the closest traffic light to the current pose

        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        light_wp = helper.next_waypoint_index_kdtree(pose, self.traffic_lights_kdtree)
        return light_wp

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self, light):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        light_wp = -1
        if self.pose:
            light_wp = self.get_closest_traffic_light(self.pose.pose)

        # Find the closest visible traffic light (if one exists)
        if light:
            state = self.get_light_state(light)
            return light_wp, state

        self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
