import rospy

from styx_msgs.msg import TrafficLight
from opencv_detector import recognize_traffic_lights

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
            rospy.loginfo("Found Traffic Light: %s", traffic_light)

        return TrafficLight.UNKNOWN
