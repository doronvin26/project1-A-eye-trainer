import streamlit as st
import cv2
import tempfile

st.title("A-EYE TRAINER: Frame Counter")

# Receive the video file
# Correct way: Just a label for the button
uploaded_video = st.file_uploader("Upload your workout video", type=["mp4", "mov", "avi"])
if uploaded_video is not None:
    # Save the file to a temp location so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    # Open the video
    cap = cv2.VideoCapture(tfile.name)
    
    frame_count = 0
    st.write("Processing frames...")
    
    # Loop through the video until it ends
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_count += 1
    
    cap.release()
    st.success(f"Done! The video has {frame_count} total frames.")