import streamlit as st
import utils
import cv2
import numpy
import io

import PIL
from camera_input_live import camera_input_live

st.set_page_config(
    page_title="Hello Text Detection",
    page_icon=":sun_with_face:",
    layout="centered",
    initial_sidebar_state="expanded",)

st.title("Hello Text Dection :sun_with_face:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20))/100

input = None 
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))
    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
        boxes, resized_image = utils.predict_image(uploaded_image_cv, conf_threshold = conf_threshold)
        result_image = utils.convert_result_to_image(uploaded_image_cv, resized_image, boxes, conf_labels=False)
        st.image(result_image, channels = "RGB")
        st.markdown(f"<h4 style='color: blue;'><strong>The result of running the AI inference on an image.</strong></h4>", unsafe_allow_html=True)
    else: 
        st.image("data/intel_rnb.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image." )

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)
    fps = camera.get(cv2.CAP_PROP_FPS)
    temp_file_2 = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
    video_row=[]

    # frame
    total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0
    
    st_frame = st.empty()
    while(camera.isOpened()):
        ret, frame = camera.read()
        if ret:
            try:
                visualized_image = utils.predict_image(frame, conf_threshold)
            except:
                visualized_image = frame
            st_frame.image(visualized_image, channels = "BGR")
            video_row.append(cv2.cvtColor(visualized_image,cv2.COLOR_BGR2RGB))  
            frame_count +=1 
            progress_bar.progress(frame_count/total_frames, text=None)
        else:
            progress_bar.empty()
            camera.release()
            st_frame.empty()
            break
            
temporary_location = None
if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an video.", type=("mp4"))
  
    if input is not None:
        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4" 
        with open(temporary_location, "wb") as out: 
            out.write(g.read())
        out.close() 

    if temporary_location is not None:
        play_video(temporary_location)
    else:
        st.video("data/sample_video.mp4")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an video." )


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




    clip = mpy.ImageSequenceClip(video_row, fps = fps)
    clip.write_videofile(temp_file_2.name)
    
    st.video(temp_file_2.name)
