import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load model
model = YOLO("trained_model.pt")

# Pascal VOC class names
PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]
class_name_to_id = {v: k for k, v in model.names.items()}
pascal_class_ids = [class_name_to_id[c] for c in PASCAL_CLASSES if c in class_name_to_id]

st.title("🚀 YOLOv8 Object Detection")
st.write("Upload an image or video to see object detection in action!")

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov"])

def draw_boxes(image, results):
    for box in results.boxes:
        cls_id = int(box.cls)
        if cls_id in pascal_class_ids:
            conf = box.conf.cpu().item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            label = f"{model.names[cls_id]}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return image

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        results = model(img_np)[0]
        img_np = draw_boxes(img_np, results)
        st.image(img_np, caption="Detected Image", use_column_width=True)

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            frame = draw_boxes(frame, results)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        cap.release()
        os.remove(video_path)
