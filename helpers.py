def check_pkg_versions():
    import streamlit_webrtc
    import cv2
    import numpy as np
    import mediapipe as mp
    import av

    print("streamlit_webrtc:", streamlit_webrtc.__version__)    
    print("cv2:",cv2.__version__)    
    print("numpy:", np.__version__)    
    print("mediapipe:", mp.__version__)    
    print("av:", av.__version__)