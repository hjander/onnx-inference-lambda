from PIL import Image
from tests import *
import pytest

from handler import OnnxModelInference


def test_inference():

    image = Image.open("7214.jpg")
    print(image)

    inference = OnnxModelInference('../densenet121/model.onnx', '../densenet121/imagenet_class_index.json')
    print(inference.infer(image))
