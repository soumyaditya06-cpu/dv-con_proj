from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from task_guided_selector import load_mobileclip, find_best_object


def run_pipeline(img_path, task, yolo_conf=0.25):
    # Load models
    model_yolo = YOLO('yolov8n.pt')
    model, tokenizer, preprocess, device = load_mobileclip()

    # Load image
    img = Image.open(img_path).convert("RGB")

    # YOLO detection
    results = model_yolo(img_path, conf=yolo_conf)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names

    detections = []

    for box, cls in zip(boxes, classes):
        detections.append({
            "class": names[int(cls)],
            "bbox": box.astype(int).tolist()
        })

    # MobileCLIP selection
    best_obj, scores, best_crop = find_best_object(
        img,
        detections,
        task,
        model,
        tokenizer,
        preprocess,
        device
    )

    print("Detected object scores:")
    for det, score in zip(detections, scores):
        print(f"{det['class']}: {score.item():.4f}")

    # Draw selected object
    x1, y1, x2, y2 = best_obj["bbox"]

    result_img = img.copy()
    draw = ImageDraw.Draw(result_img)
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=5)

    plt.figure(figsize=(8, 6))
    plt.imshow(result_img)
    plt.axis("off")
    plt.show()

    return best_obj, scores, result_img


if __name__ == "__main__":
    img_path = "/content/drive/MyDrive/coco_project/datasets/val2014/COCO_val2014_000000000139.jpg"
    task = "a vase for flowers"

    run_pipeline(img_path, task)
