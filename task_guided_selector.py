import torch
from mobileclip import create_model_and_transforms, get_tokenizer


TASK_PRIORS = {
    "an object suitable to step on safely": ["chair", "bench"],

    "an object suitable for sitting comfortably": ["chair", "couch", "bench"],

    "an object suitable for holding flowers": ["vase", "bottle", "cup"],

    "a tool used to remove hot food from fire": ["spoon", "fork", "tongs"],

    "an object used to water a plant": ["bottle", "cup", "watering can"],

    "a utensil used to remove lemon from tea": ["spoon", "fork", "cup"],

    "a tool used for digging a hole": ["shovel", "spoon"],

    "a tool used to open a bottle": ["bottle opener", "scissors", "knife"],

    "a tool used to open a parcel": ["scissors", "knife"],

    "an object used to serve wine": ["wine glass", "cup", "bottle"],

    "an object used to pour sugar": ["cup", "bowl", "spoon"],

    "a kitchen utensil used to spread butter": ["knife", "spoon", "scissors"],

    "an object used to extinguish fire": ["fire extinguisher", "bottle"],

    "an object used to beat or clean a carpet": ["stick", "bat", "chair"]
}

BOOST_FACTOR = 3


def load_mobileclip(checkpoint_path="/content/dv-con_proj/mobileclip_s1.pt"):
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

    if len(cropped_images) == 0:
        return None, torch.tensor([]), None

    text_tokens = tokenizer([task]).to(device)

    image_inputs = torch.stack([
        preprocess(crop) for crop in cropped_images
    ]).to(device)

    with torch.no_grad():

        image_features, text_features, logit_scale = model(
            image=image_inputs,
            text=text_tokens
        )

        logits = logit_scale * image_features @ text_features.T

        probs = torch.softmax(logits.squeeze(), dim=0)

    if task in TASK_PRIORS:

        preferred = TASK_PRIORS[task]

        for i, det in enumerate(detections):

            detected_class = det["class"].lower()

            if detected_class in preferred:
                probs[i] *= BOOST_FACTOR

        probs = probs / probs.sum()

    best_idx = torch.argmax(probs).item()

    return detections[best_idx], probs, cropped_images[best_idx]
