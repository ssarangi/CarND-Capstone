#import rospy

#from styx_msgs.msg import TrafficLight
from opencv_detector import recognize_traffic_lights
from dl_detector import DeepLearningDetector
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
    def __init__(self):
        self.dl_classifier = DeepLearningDetector()
        self.use_opencv = False
        self.use_dl = True

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        p = Pool(processes=2)
        cv_traffic_lt = p.map(recognize_traffic_lights, [image])
        dl_traffic_lt = p.map(self.dl_classifier.detect, [image])
        p.close()

        detected_light = TrafficLight.UNKNOWN
               
        if cv_traffic_lt == dl_traffic_lt:
            detected_light = cv_traffic_lt           
        elif dl_traffic_lt == TrafficLight.UNKNOWN and cv_traffic_lt != TrafficLight.UNKNOWN:
             detected_light = cv_traffic_lt
        elif cv_traffic_lt == TrafficLight.UNKNOWN and dl_traffic_lt != TrafficLight.UNKNOWN:
            detected_light = dl_traffic_lt

        #rospy.logdebug("Found Traffic Light: %s", traffic_light_msg_to_string_dl(detected_light))
        
        return detected_light
        
        '''
        if self.use_opencv:
            traffic_light = recognize_traffic_lights(image)
            rospy.logdebug("Found Traffic Light: %s", traffic_light_msg_to_string(traffic_light))
            return traffic_light

        if self.use_dl:
            predicted_label = self.dl_classifier.detect(image)
            rospy.logdebug("Found Traffic Light: %s", traffic_light_msg_to_string_dl(predicted_label))
            return predicted_label
        return TrafficLight.UNKNOWN
        '''
