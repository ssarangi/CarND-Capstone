import numpy as np
import tensorflow as tf
from utils import label_map_util
import rospy
import os
import time

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

THRESHOLD = 0.3

class DeepLearningDetector:
  def __init__(self):
    self.carla = True
    if self.carla:
      PATH_TO_CKPT = './light_classification/model/frozen_inference_graph.pb'
      PATH_TO_LABELS = './light_classification/model/boschlabel.pbtxt'
      self.NUM_CLASSES = 14
    else:
      PATH_TO_CKPT = 'models/sim_frozen_inference_graph.pb'
      PATH_TO_LABELS = 'models/object-detection.pbtxt'
      self.NUM_CLASSES = 4

    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES,
                                                                use_display_name=True)
    rospy.logwarn(categories)
    self.category_index = label_map_util.create_category_index(categories)

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

        rospy.loginfo("nums = {}, scores_sum = {}".format(nums, scores_sum))

        maxn = np.max(nums)
        cands = (nums == maxn)
        prediction = np.argmax(scores_sum * cands)

        rospy.loginfo("prediction = {}, max_score = {}".format(prediction, scores[0]))
        rospy.loginfo("prediction time = {:.4f} s".format(time.time() - t0))
        # return prediction
        return 0