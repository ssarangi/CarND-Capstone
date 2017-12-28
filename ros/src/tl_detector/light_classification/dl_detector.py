import numpy as np
import tensorflow as tf
import rospy
import time
from styx_msgs.msg import TrafficLight

from utils import load_pbtxt

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

THRESHOLD = 0.3

class DeepLearningDetector:
  def __init__(self):
    rospy.logwarn('DeepLearningDetector init method')
    self.carla = True
    if self.carla:
      PATH_TO_CKPT = './light_classification/model/carla_frozen_inference_graph.pb'
      PATH_TO_LABELS = './light_classification/model/boschlabel.pbtxt'
      self.NUM_CLASSES = 14
    else:
      PATH_TO_CKPT = './light_classification/model/sim_frozen_inference_graph.pb'
      PATH_TO_LABELS = './light_classification/model/object-detection.pbtxt'
      self.NUM_CLASSES = 4

    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    self.category_index = load_pbtxt.get_labels(PATH_TO_LABELS, self.NUM_CLASSES)
    self.labels_2_traffic_light_id = DeepLearningDetector.category_index_to_traffic_light(self.category_index)

  @classmethod
  def category_index_to_traffic_light(cls, category_index):
    new_category_index = {}
    for item_id, item in category_index.items():
      item_name = str(item['name']).lower()
      if 'red' in item_name:
        new_category_index[item_id] = TrafficLight.RED
      elif 'yellow' in item_name:
        new_category_index[item_id] = TrafficLight.YELLOW
      elif 'green' in item_name:
        new_category_index[item_id] = TrafficLight.GREEN
      elif 'off' in item_name:
        new_category_index[item_id] = TrafficLight.UNKNOWN

    return new_category_index

  def detect(self, image_np):
    t0 = time.time()
    with self.detection_graph.as_default():
      with tf.Session(graph=self.detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # Counts classes
        nums = np.zeros(self.NUM_CLASSES, dtype=np.uint8)
        scores_sum = np.zeros(self.NUM_CLASSES)
        for tlk in range(len(classes)):
          c = self.category_index[int(classes[tlk])]
          s = scores[tlk]
          if s > THRESHOLD:
            nums[c['id']] += 1
            scores_sum[c['id']] += s

        maxn = np.max(nums)
        cands = (nums == maxn)
        prediction = np.argmax(scores_sum * cands)

        actual_prediction = self.labels_2_traffic_light_id.get(prediction, TrafficLight.UNKNOWN)
        return actual_prediction


def process_top_level_instance(instance_obj, image):
  return instance_obj.detect(image)


def another_method(ss):
  return TrafficLight.RED

if __name__ == '__main__':
  import cv2
  img = cv2.imread('model/324.png')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  dl = DeepLearningDetector()
  dl.detect(img)