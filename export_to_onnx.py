from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("D:\\YOLO\\ultralytics-main\\best.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo26n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("D:\\YOLO\\ultralytics-main\\uav-yolo11n.onnx")

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")