import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe pose and face mesh models
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)

# get directory for YOLO files
file_path = os.path.realpath(__file__)
src_dir = os.path.dirname(file_path)
yolo_dir = src_dir.replace("src", "YOLOv4-tiny")

# Load YOLOv4-tiny class names
with open(os.path.join(yolo_dir, "coco.names"), "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load YOLOv4-tiny model
net = cv2.dnn.readNet(os.path.join(yolo_dir, "yolov4-tiny.weights"),
                      os.path.join(yolo_dir, "yolov4-tiny.cfg")
                     )

# Use CPU
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get YOLOv4-tiny output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# resize frames to HD while maintaining aspect ratio
resize = True

# Set max width and height
MAX_WIDTH = 1280
MAX_HEIGHT = 720

# Webcam feed
cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if not ret:
    raise Exception("Unable to connect to camera")

# Flip for selfie mode and convert to RGB
# frame = cv2.flip(frame, 1)

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

# Create Output folder to contain recorded footage
if not os.path.exists("Output"):
    os.makedirs("Output")

# Initiate VideoWriter
video_writer = cv2.VideoWriter("Output/processed_footage.mp4", 
                               cv2.VideoWriter_fourcc(*"MP4V"),
                               10, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
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

    # Create a blob from input frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # create variables to hold bounding box info for phones detected
    phone = {"x": [], "y": [], "width": [], "height": []}

    # Loop over detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Look for "cell phone" class
            if class_names[class_id] == "cell phone" and confidence > 0.5:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                phone["width"].append(int(detection[2] * w))
                phone["height"].append(int(detection[3] * h))

                phone["x"].append(int(center_x - phone["width"][-1] / 2))
                phone["y"].append(int(center_y - phone["height"][-1] / 2))
    
    # Convert frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with FaceMesh and Pose
    face_results = face_mesh.process(rgb)

    hor_dir = "Forward"  # Default horizontal direction
    vert_dir = "Forward" # Default vertical direction

    # create varibles to hold bounding box and anomaly info for detected faces
    person = {"x1": [], "x2": [], "y1": [], "y2": [],
              "hor_dir": [], "speech": [], "color": []
             }

    # ---- FACE BOX LOGIC (FACEMESH) ----
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Get face landmarks
            lms = face_landmarks.landmark
            
            # Get x and y coordinates of facial landmarks
            x1 = int(min([lm.x for lm in lms]) * w)
            x2 = int(max([lm.x for lm in lms]) * w)
            y1 = int(min([lm.y for lm in lms]) * h)
            y2 = int(max([lm.y for lm in lms]) * h)
            
            # ---- HEAD DIRECTION LOGIC (POSE) ----
            pose_x1 = int(max(0, x1 - 0.2*w))
            pose_y1 = int(max(0, y1 - 0.15*h))
            pose_x2 = int(min(w, x2 + 0.2*w))
            pose_y2 = int(min(h, y2 + 0.5*h))
            pose_results = pose.process(rgb[pose_y1:pose_y2, pose_x1:pose_x2])

            anomaly = False
            speech = ""
            
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

                # Horizontal movement (Left/Right)
                hor_orient = (nose.x - shoulder_center_x) / nose.z
                if hor_orient > 0.03:
                    hor_dir = "Right"
                    anomaly = True
                elif hor_orient < -0.03:
                    hor_dir = "Left"
                    anomaly = True
                else:
                    hor_orient = ""
                    
                # Check for speech
                lips_movement = (lms[15].y - lms[13].y) / abs(nose.z)
                if lips_movement > 0.017:
                    speech = "Talking"
                    anomaly = True
                else:
                    speech = ""

            # Pick box color based on anomalt detection
            if anomaly:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            # add box and anomaly info for the current person
            person["x1"].append(x1)
            person["x2"].append(x2)
            person["y1"].append(y1)
            person["y2"].append(y2)
            person["color"].append(color)
            person["hor_dir"].append(hor_dir)
            person["speech"].append(speech)

            # Draw face rectangle
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, f"{hor_dir} | {speech}", (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    # create a flipped version of frame
    frame_copy = frame.copy()

    # add bounding box and anomaly info for detected people
    for x1, y1, x2, y2, color, hor_dir, speech in zip(person["x1"], person["y1"], person["x2"], person["y2"],
                                                      person["color"], person["hor_dir"], person["speech"]):
        # Draw face rectangle (face box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw face rectangle (face box)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

        # Add text to face box
        cv2.putText(frame, f"{hor_dir} | {speech}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # add bounding box to detected phones
    for phone_x, phone_y, phone_w, phone_h in zip(phone["x"], phone["y"], phone["width"], phone["height"]):
        cv2.rectangle(frame,(phone_x, phone_y),
                      (phone_x + phone_w, phone_y + phone_h),
                      (0, 0, 255), 2)
        
        cv2.rectangle(frame_copy,(phone_x, phone_y),
                      (phone_x + phone_w, phone_y + phone_h),
                      (0, 0, 255), 2)

        cv2.putText(frame, "Phone", (phone_x, phone_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # add frame to video output
    video_writer.write(frame)

    # Flip for selfie mode
    frame_copy = cv2.flip(frame_copy, 1)

     # add bounding box and anomaly info for detected people in streaming video
    for x1, y1, x2, y2, color, hor_dir, speech in zip(person["x1"], person["y1"], person["x2"], person["y2"],
                                                      person["color"], person["hor_dir"], person["speech"]):
        cv2.putText(frame_copy, f"{hor_dir} | {speech}", (w - x2, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    # add bounding box to detected phones in streaming video
    for phone_x, phone_y, phone_w, phone_h in zip(phone["x"], phone["y"], phone["width"], phone["height"]):
        cv2.putText(frame_copy, "Phone", (w - phone_x - phone_w, phone_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Show result
    cv2.imshow("Examination Anomaly Detection", frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
