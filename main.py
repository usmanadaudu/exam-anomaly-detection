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

webrtc_streamer(key="key")