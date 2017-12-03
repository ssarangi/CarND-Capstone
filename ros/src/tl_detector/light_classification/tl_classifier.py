import rospy

from styx_msgs.msg import TrafficLight
from opencv_detector import recognize_traffic_lights


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
    def __init__(self):
        self.use_opencv = True

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.use_opencv:
            traffic_light = recognize_traffic_lights(image)
            rospy.logdebug("Found Traffic Light: %s", traffic_light_msg_to_string(traffic_light))
            return traffic_light

        return TrafficLight.UNKNOWN
