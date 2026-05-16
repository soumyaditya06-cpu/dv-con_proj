from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
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

    # Grad-CAM
    target_layer = model.image_encoder.model.conv_exp

    cam = GradCAM(
        model=model.image_encoder,
        target_layers=[target_layer]
    )

    input_tensor = preprocess(best_crop).unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = np.array(best_crop.resize((256, 256))).astype(np.float32) / 255.0

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Overlay CAM on original image
    x1, y1, x2, y2 = best_obj["bbox"]

    cam_resized = Image.fromarray(visualization).resize((x2 - x1, y2 - y1))

    original_with_cam = img.copy()
    original_with_cam.paste(cam_resized, (x1, y1))

    draw = ImageDraw.Draw(original_with_cam)
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=5)

    plt.figure(figsize=(8, 6))
    plt.imshow(original_with_cam)
    plt.axis("off")
    plt.show()

    return best_obj, scores, original_with_cam


if __name__ == "__main__":
    img_path = "/content/drive/MyDrive/coco_project/datasets/val2014/COCO_val2014_000000000139.jpg"
    task = "a vase for flowers"

    run_pipeline(img_path, task)
