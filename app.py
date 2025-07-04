import streamlit as st
import cv2
import torch
import tempfile
from ultralytics import YOLO
import numpy as np
import joblib 

model = YOLO('best.pt')

def detect_animals_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in ['tiger']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")
    cap.release()

def detect_animals_image(image):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model.predict(frame)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in ['tiger']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, channels="RGB")

st.title("Animal Detection using YOLOv8")

option = st.radio("Choose input type", ['Image', 'Video'])

if option == 'Image':
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = uploaded_image.read()
        detect_animals_image(st.image(uploaded_image).image)

elif option == 'Video':
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        detect_animals_video(tfile.name)
