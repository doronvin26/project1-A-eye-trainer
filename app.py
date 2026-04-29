import streamlit as st
import cv2
import tempfile
import time

st.title("A-EYE TRAINER: Video Preview")

uploaded_video = st.file_uploader("Upload a video to analyze", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Create a placeholder in the Streamlit UI for the video frames
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # OpenCV uses BGR, but Streamlit/Web needs RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update the image in the browser
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        frame_count += 1
        status_text.text(f"Current Frame: {frame_count}")
        
        # Short sleep to make the playback look natural (optional)
        # time.sleep(0.01) 

    cap.release()
    st.success(f"Processing complete! Total frames: {frame_count}")