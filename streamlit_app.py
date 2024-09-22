import streamlit as st
import utils
import cv2
import numpy
import io
import tempfile
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

import cv2
import tempfile
import io
import streamlit as st

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)
    fps = camera.get(cv2.CAP_PROP_FPS)
    temp_file_2 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_row = []
    # 전체 프레임 수 계산
    total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0
    st_frame = st.empty()
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            try:
                # 예측 모델 함수 (정의되지 않은 상태로 가정)
                visualized_image = utils.predict_image(frame, conf_threshold)
            except:
                visualized_image = frame
            # BGR -> RGB로 변환 후 255로 나눠서 [0, 1] 범위로 맞춤
            visualized_image = cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB) / 255.0
            # Streamlit에 이미지 표시
            st_frame.image(visualized_image)
            # 비디오 행렬에 변환된 프레임 추가
            video_row.append(visualized_image)
            # 진행 상황 표시
            frame_count += 1
            progress_bar.progress(frame_count / total_frames, text=None)
        else:
            progress_bar.empty()
            camera.release()
            st_frame.empty()
            break

# 파일 업로드 처리
temporary_location = None
if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input_file = st.sidebar.file_uploader("Choose a video.", type=("mp4"))
    if input_file is not None:
        # 파일을 임시 경로에 저장
        g = io.BytesIO(input_file.read())
        temporary_location = "upload.mp4"
        with open(temporary_location, "wb") as out:
            out.write(g.read())
        out.close()
    # 업로드된 비디오 파일이 있는 경우 비디오 재생
    if temporary_location is not None:
        play_video(temporary_location)
    else:
        st.video("data/sample_video.mp4")
        st.write("Click on 'Browse Files' in the sidebar to run inference on a video.")

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
