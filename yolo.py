from ultralytics import YOLO
from PIL import Image

model = YOLO('yolov8n.pt')

img_path = '/content/drive/MyDrive/coco_project/datasets/val2014/COCO_val2014_000000000139.jpg'

results = model(img_path, conf=0.25)

results[0].show()

# Extract detections
boxes = results[0].boxes.xyxy.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()
scores = results[0].boxes.conf.cpu().numpy()
names = results[0].names

print([(names[int(c)], s) for c, s in zip(classes, scores)])
