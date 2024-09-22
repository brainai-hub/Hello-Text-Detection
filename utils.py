import openvino as ov
import cv2
import numpy as np
from pathlib import Path

core = ov.Core()
model = core.read_model(model='models/v3-small_224_1.0_float.xml')
compiled_model = core.compile_model(model = model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def preprocess(image, input_layer):
    input_image = cv2.resize(src=image, dsize=(224, 224))
    input_image = np.expand_dims(input_image, 0)
    return input_image 

def predict_image(image, conf_threshold):
    input_image = preprocess(image, input_layer)
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)
    imagenet_filename = Path('data/imagenet_2012.txt')
    imagenet_classes = imagenet_filename.read_text().splitlines()
    imagenet_classes = ["background"] + imagenet_classes
    imagenet_classes = imagenet_classes[result_index]
    return imagenet_classes
