import tensorflow as tf
import numpy as np

# Reference: https://github.com/udacity/CarND-Object-Detection-Lab
def Get_Predicted_Label(image, simulation=True):
    if simulation:
        graph_filename = 'model/sim_frozen_inference_graph.pb'
    else:
        graph_filename = 'model/carla_frozen_inference_graph.pb'
    labels_filename = 'model/object-detection.pbtxt'
    labels = load_labels(labels_filename)

    labels_dic = {'green':1,
                       'red':2,
                       'yellow':3,
                       'none':4}

    print("Initializing TensorFlow...")
    detection_graph = tf.Graph()
    # configure for a GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # load trained tensorflow graph
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_filename, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

            sess = tf.Session(graph=detection_graph, config=config)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            image_np = load_image_into_numpy_array(image)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
                 feed_dict={image_tensor: image_np_expanded})

            # Remove unnecessary dimensions
            class_  = np.int32(np.squeeze(classes).tolist())

            index = next((i for i, clsid in enumerate(class_) if clsid < 4), None)

            return index




def labels(labels_filename):
    return [line.rstrip() for line in tf.gfile.GFile(labels_filename)]

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
