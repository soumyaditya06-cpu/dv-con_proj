import torch
from mobileclip import create_model_and_transforms, get_tokenizer


def load_mobileclip(checkpoint_path="mobileclip_s1.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = create_model_and_transforms(
        model_name='mobileclip_s1',
        pretrained=checkpoint_path,
        device=device
    )

    tokenizer = get_tokenizer('mobileclip_s1')
    model.eval()

    return model, tokenizer, preprocess, device


def find_best_object(img, detections, task, model, tokenizer, preprocess, device):
    cropped_images = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = img.crop((x1, y1, x2, y2))
        cropped_images.append(crop)

    text_tokens = tokenizer([task]).to(device)
    image_inputs = torch.stack([preprocess(crop) for crop in cropped_images]).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(
            image=image_inputs,
            text=text_tokens
        )

        logits = logit_scale * image_features @ text_features.T
        probs = torch.softmax(logits.squeeze(), dim=0)

    best_idx = torch.argmax(probs).item()

    return detections[best_idx], probs, cropped_images[best_idx]
