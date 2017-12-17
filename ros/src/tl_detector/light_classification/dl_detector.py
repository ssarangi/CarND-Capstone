

class DeepLearningDetector:
  def __init__(self):
    PATH_TO_CKPT = 'models/frozen_inference_graph.pb'
    PATH_TO_LABELS = 'models/boschlabel.pbtxt'