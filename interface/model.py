import clip
from PIL import Image
import torch

CLIP_MODEL = 'ViT-L/14@336px'

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load(CLIP_MODEL, device)

def encode_image(img_filename):
    with Image.open(img_filename) as img:
        image_input = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input).numpy()
        image_features = image_features.flatten()
    return image_features

def encode_text(text):
    text_inputs = torch.Tensor(clip.tokenize(text)).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_inputs).cpu().numpy()
        return text_embeddings
