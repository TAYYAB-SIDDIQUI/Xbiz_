from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
model = YOLO(model_path)

results = model.predict(r"E:\Xbiz_assignment\OCR_segmentaiton\static\docs\aadhhar.png", save=True) # saves the result in runs/detect/predict
