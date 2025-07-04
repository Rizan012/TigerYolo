# 🐅 TigerYolo

A simple Streamlit app that uses **YOLOv8** to detect tigers in uploaded videos and images.

---

## Features

- Upload **video** files (`.mp4`, `.avi`, `.mov`, `.mkv`)
- Upload **image** files (`.jpg`, `.jpeg`, `.png`)
- Detects **tigers** using a custom-trained YOLOv8 model
- Displays bounding boxes and confidence scores in real-time
- Image dataset was manually labeled using **Roboflow**

---

## Tech Stack

- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- Python

---

## Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
