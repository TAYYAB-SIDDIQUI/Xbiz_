import torch
import clip
from PIL import Image
import os

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Labels
labels = ["circle", "square", "triangle", "star", "rectangle","pentagon","paralellogram","hexagon","trapezoid"]
text = clip.tokenize(labels).to(device)

# Folder with images
folder_path = "/content/drive/MyDrive/Test"

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        # Load and preprocess image
        image_path = os.path.join(folder_path, filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Map labels to probabilities
        pred_dict = dict(zip(labels, probs[0]))

        # Print results
        top_label = labels[probs[0].argmax()]
        top_prob = probs[0].max() * 100
        if top_prob<30:
          top_label="undetected"
        print(f"Image: {filename}")
        print(f"Predicted: {top_label} ({top_prob:.2f}%)")
        print("All probabilities:", pred_dict)
        print("-" * 40)
