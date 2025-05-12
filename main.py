import cv2
import mediapipe as mp

# Initialize MediaPipe pose and face mesh models
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)

# resize frames to HD while maintaining aspect ratio
resize = True

# Set max width and height
MAX_WIDTH = 1280
MAX_HEIGHT = 720

# Webcam feed
cap = cv2.VideoCapture(0)

ret, frame = cap.read()

# Flip for selfie mode and convert to RGB
frame = cv2.flip(frame, 1)

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

# Initiate VideoWriter
video_writer = cv2.VideoWriter("processed_footage.mp4", 
                               cv2.VideoWriter_fourcc(*"MP4V"),
                               10, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip for selfie mode and convert to RGB
    frame = cv2.flip(frame, 1)
    
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
    
    # Convert frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with FaceMesh and Pose
    face_results = face_mesh.process(rgb)

    hor_dir = "Forward"  # Default horizontal direction
    vert_dir = "Forward" # Default vertical direction

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
                    hor_dir = "Left"
                    anomaly = True
                elif hor_orient < -0.03:
                    hor_dir = "Right"
                    anomaly = True
                else:
                    hor_orient = ""
                    
                # Check for speech
                lips_movement = (lms[15].y - lms[13].y) / abs(nose.z)
                if lips_movement > 0.015:
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{hor_dir} | {speech}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # add frame to video output
    video_writer.write(frame)
    
    # Show result
    cv2.imshow("Examination Anomaly Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
