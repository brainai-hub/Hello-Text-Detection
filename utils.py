import openvino as ov
import cv2
import numpy as np
from pathlib import Path

core = ov.Core()
model = core.read_model(model='./model/horizontal-text-detection-0001.xml')
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





core = ov.Core()

model_xml_path = "./model/horizontal-text-detection-0001.xml"
model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output("boxes")


# ## Load an Image
# [back to top ⬆️](#Table-of-contents:)
# 

# In[8]:


import cv2

# 이미지 파일 경로
image_filename = "./data/intel_rnb.jpg"
# 이미지를 BGR 포맷으로 읽기
image = cv2.imread(image_filename)
# 이미지 창에 표시 (cv2는 BGR 포맷을 사용하므로 변환할 필요 없음)
cv2.imshow('Image', image)
# 키 입력을 기다림 (0은 무한 대기)
cv2.waitKey(0)
# 모든 창 닫기
cv2.destroyAllWindows()


# In[9]:


# N,C,H,W = batch size, number of channels, height, width.
N, C, H, W = input_layer_ir.shape
# Resize the image to meet network expected input sizes.
resized_image = cv2.resize(image, (W, H))
# Reshape to the network input shape.
input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)


# ## Do Inference
# [back to top ⬆️](#Table-of-contents:)
# 

# In[10]:


# Create an inference request.
boxes = compiled_model([input_image])[output_layer_ir]

# Remove zero only boxes.
boxes = boxes[~np.all(boxes == 0, axis=1)]


# ## Visualize Results
# [back to top ⬆️](#Table-of-contents:)
# 

# In[11]:


# For each detection, the description is in the [x_min, y_min, x_max, y_max, conf] format:
# The image passed here is in BGR format with changed width and height. To display it in colors expected by matplotlib, use cvtColor function
def convert_result_to_image(bgr_image, resized_image, boxes, threshold=0.3, conf_labels=True):
    # Define colors for boxes and descriptions.
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}

    # Fetch the image shapes to calculate a ratio.
    (real_y, real_x), (resized_y, resized_x) = (
        bgr_image.shape[:2],
        resized_image.shape[:2],
    )
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Convert the base image from BGR to RGB format.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Iterate through non-zero boxes.
    for box in boxes:
        # Pick a confidence factor from the last place in an array.
        conf = box[-1]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio.
            # If the bounding box is found at the top of the image,
            # position the upper box bar little lower to make it visible on the image.
            (x_min, y_min, x_max, y_max) = [
                (int(max(corner_position * ratio_y, 10)) if idx % 2 else int(corner_position * ratio_x)) for idx, corner_position in enumerate(box[:-1])
            ]

            # Draw a box based on the position, parameters in rectangle function are: image, start_point, end_point, color, thickness.
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)

            # Add text to the image based on position and confidence.
            # Parameters in text function are: image, text, bottom-left_corner_textfield, font, font_scale, color, thickness, line_type.
            if conf_labels:
                rgb_image = cv2.putText(
                    rgb_image,
                    f"{conf:.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    1,
                    cv2.LINE_AA,
                )

    return rgb_image


# In[8]:


image = convert_result_to_image(image, resized_image, boxes, conf_labels=False)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


get_ipython().system(' jupyter nbconvert --to script Steamlit_app.ipynb')


# In[ ]:
