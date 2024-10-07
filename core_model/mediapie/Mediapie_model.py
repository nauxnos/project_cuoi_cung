import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
from torch import nn

class Mediapie_Hand(nn.Module):
    def __init__(self, weight_path):
        super(Mediapie_Hand, self).__init__()
        self.base_option = python.BaseOptions(model_asset_path=weight_path)
        self.option = vision.HandLandmarkerOptions(base_options=self.base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def forward(self, x):
        return self.detector.detect(x)

