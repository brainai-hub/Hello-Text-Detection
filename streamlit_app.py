import streamlit as st
import utils
import cv2
import numpy
import PIL
from camera_input_live import camera_input_live

st.set_page_config(
    page_title="Hello Image Classification",
    page_icon=":sun_with_face:",
    layout="centered",
    initial_sidebar_state="expanded",)

st.title("Hello Image Classification :sun_with_face:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20))/100

input = None 
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))
    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
        imagenet_classes = utils.predict_image(uploaded_image_cv, conf_threshold = conf_threshold)
        st.image(uploaded_image_cv, channels = "BGR")
        st.markdown(f"<h2 style='color: red;'>{''.join(imagenet_classes)}</h2><h4 style='color: blue;'><strong>The result of running the AI inference on an image:</strong></h4>", unsafe_allow_html=True)
    else: 
        st.image("data/coco.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image." )
        
if source_radio == "WEBCAM":
    st.sidebar.header("My Webcam Test")
    image = camera_input_live()
    # Ensure the image is valid
    if image is not None:
        uploaded_image = PIL.Image.open(image)
        uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
        # Display the webcam image
        st.image(uploaded_image_cv, channels="BGR")  
        # Get the image dimensions
        height, width, _ = uploaded_image_cv.shape
        # Display the dimensions in the sidebar
        st.sidebar.write(f"Webcam image size: {width} x {height} pixels")
    else:
        st.sidebar.error("No image captured from webcam.")
