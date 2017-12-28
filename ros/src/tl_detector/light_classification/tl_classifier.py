import functools
import rospy
import time

from styx_msgs.msg import TrafficLight
from opencv_detector import recognize_traffic_lights
from dl_detector import DeepLearningDetector, process_top_level_instance, another_method
from multiprocessing import Pool


def traffic_light_msg_to_string_dl(traffic_light_msg):
    if traffic_light_msg == 1:
        return 'GREEN'
    elif traffic_light_msg == 2:
        return 'RED'
    elif traffic_light_msg == 3:
        return 'YELLOW'
    elif traffic_light_msg == 4:
        return 'OFF'


def traffic_light_msg_to_string(traffic_light_msg):
    # UNKNOWN = 4
    # GREEN = 2
    # YELLOW = 1
    # RED = 0
    if traffic_light_msg == 0:
        return 'RED'
    elif traffic_light_msg == 1:
        return 'YELLOW'
    elif traffic_light_msg == 2:
        return 'GREEN'
    elif traffic_light_msg == 4:
        return 'UNKNOWN'


class TLClassifier(object):
    def __init__(self, is_carla):
        self.use_opencv = False
        self.use_dl = True
        self.is_carla = is_carla
        if is_carla:
            self.dl_classifier = DeepLearningDetector()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        rospy.logdebug('Carla Flag')
        if not self.is_carla:
            detected_light = recognize_traffic_lights(image)
        else:
            detected_light = self.dl_classifier.detect(image)

        rospy.logdebug("Found Traffic Light: %s", traffic_light_msg_to_string_dl(detected_light))
        return detected_light