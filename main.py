# import cv2
# import streamlit as st

# st.title("Webcam Live Feed")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# # camera = st.camera_input("Cheking camera input", disabled=not run)

# while run:
#     for i in range(1):
#         camera = cv2.VideoCapture(i)
#         if camera.isOpened():
#             print(f"Input {i} is a valid camera value for VIDEO_SOURCE")
#             ret, frame = camera.read()

#             # Do nothing when camera frame cannot be read
#             if not ret:
#                 break

#             # Flip the frame to show live cam properly
#             frame = cv2.flip(frame, 1)

#             # Convert frame from BGR to RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Show frame in webapp
#             FRAME_WINDOW.image(frame)
#         else:
#             st.write("Camera returned no frame...")
# else:
#     st.write('Stopped')

from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import cv2
import av

# resize frames to HD while maintaining aspect ratio
resize = True

# Set max width and height
MAX_WIDTH = 1280
MAX_HEIGHT = 720

# Set maximum number of faces to track
num_faces = 5

# Initialize MediaPipe pose and face mesh models
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=num_faces)

def frame_callback(image):

    frame = image.to_ndarray(format="bgr24")
    
    # Flip webcam feed horizontally
    frame = frame[:, ::-1, :]

    # Get frame shape
    h, w, _ = frame.shape

    if resize:
        # Get new domension while maintaining aspect ratio
        scale = min(MAX_WIDTH / w, MAX_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize frame
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
        # Get new frame shape
        h, w, _ = frame.shape


    return av.VideoFrame.from_ndarray(frame, format="bgr24")

webrtc_streamer(key="webcam_footages", video_frame_callback=frame_callback)