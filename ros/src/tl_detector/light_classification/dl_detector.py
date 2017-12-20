

class DeepLearningDetector:
  def __init__(self):
    PATH_TO_CKPT = 'models/sim_frozen_inference_graph.pb'
    PATH_TO_LABELS = 'models/object-detection.pbtxt'
