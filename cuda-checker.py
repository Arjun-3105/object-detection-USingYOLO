from ultralytics import YOLO

# Load the trained model
model = YOLO("yolov3-sppu.pt")


# Print class labels
print("Class labels the model can predict:")
for idx, name in model.names.items():
    print(f"{idx}: {name}")
