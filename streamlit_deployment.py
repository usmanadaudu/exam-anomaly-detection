import streamlit as st
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
# mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose()
# mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=num_faces)

def frame_callback(image):

    print("Callback started")

    frame = image.to_ndarray(format="bgr24")
    
    # Flip webcam feed horizontally
    frame = cv2.flip(frame, 1)

    print("Image flipped")

    # # Get frame shape
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

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_img.flags.writeable = False

    # Extract faces and facial landmarks
    faces = face_mesh.process(rgb_img)

    if faces.multi_face_landmarks:
        print("face landmarks found")
        for face in faces.multi_face_landmarks:
            face_landmark = face.landmark

            # Get boundary coordinates of facial landmarks
            x1 = int(min([facial_point.x for facial_point in face_landmark]) * w)
            x2 = int(max([facial_point.x for facial_point in face_landmark]) * w)
            y1 = int(min([facial_point.y for facial_point in face_landmark]) * h)
            y2 = int(max([facial_point.y for facial_point in face_landmark]) * h)

            # Crop out current person
            pose_x1 = int(max(0, x1 - 0.2*w))
            pose_y1 = int(max(0, y1 - 0.15*h))
            pose_x2 = int(min(w, x2 + 0.2*w))
            pose_y2 = int(min(h, y2 + 0.5*h))

            print(pose_x1, pose_x2, pose_y1, pose_y2)

            rgb_img.flags.writeable = True

            print(rgb_img.shape)
            print(rgb_img[pose_y1:pose_y2, pose_x1:pose_x2, :].shape)
            print(rgb_img[pose_y1:pose_y2, pose_x1:pose_x2].shape)
            print(type(rgb_img))

            person_crop = rgb_img[pose_y1:pose_y2, pose_x1:pose_x2, :]

            # person_crop.flags.writeable = False

            # # Get pose landmarks of the current person
            # pose_result = pose.process(rgb_img[pose_y1:pose_y2, pose_x1:pose_x2, :])
            pose_result = pose.process(person_crop)

            # print("extracted pose 1")

            # rgb_img.flags.writeable = True

            # print("extracted pose 2")

            anomaly = False
            speech = ""
            hor_dir = ""

            if pose_result.pose_landmarks:
                print("Pose result found")
                pose_landmark = pose_result.pose_landmarks.landmark

                nose = pose_landmark[pose.PoseLandmark.NOSE]
                left_shoulder = pose_landmark[pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_landmark[pose.PoseLandmark.RIGHT_SHOULDER]

                shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

                # Horizontal movement (Left/Right)
                hor_orient = (nose.x - shoulder_center_x) / nose.z
                if hor_orient > 0.03:
                    hor_dir = "Left"
                    anomaly = True
                elif hor_orient < -0.03:
                    hor_dir = "Right"
                    anomaly = True
                else:
                    hor_dir = ""
                    
                # Check for speech
                lips_movement = (face_landmark[15].y - face_landmark[13].y) / abs(nose.z)
                if lips_movement > 0.007:
                    speech = "Talking"
                    anomaly = True
                else:
                    speech = ""

            # Pick box color based on anomalt detection
            if anomaly:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            # Draw face rectangle
            cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            
            cv2.putText(frame, f"{hor_dir} | {speech}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show cropped image per person
            cv2.rectangle(rgb_img, (pose_x1, pose_y1), (pose_x2, pose_y2), (0, 255, 255), 2)

    frame = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    return av.VideoFrame.from_ndarray(frame, format="bgr24")

webrtc_streamer(key="webcam_footages", video_frame_callback=frame_callback)