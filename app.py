import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from model import Generator
import urllib.request
import os

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Weights download
# -----------------------
WEIGHT_URL = "https://github.com/Mubashir214/Conditional-GAN/releases/download/v1.0.0/cuhk_best_generator.pth"
WEIGHT_PATH = "best_generator.pth"

if not os.path.exists(WEIGHT_PATH):
    st.info("Downloading model weights...")
    urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_PATH)

# -----------------------
# LOAD MODEL (FIX HERE)
# -----------------------
@st.cache_resource
def load_model():
    model = Generator().to(device)

    state_dict = torch.load(WEIGHT_PATH, map_location=device)

    # remove "module." prefix
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        cleaned_state_dict[new_key] = v

    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(cleaned_state_dict, strict=False)

    model.eval()
    return model

gen = load_model()

# -----------------------
# Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def process_image(img):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = gen(tensor).squeeze(0).cpu()

    out = (out * 0.5 + 0.5).clamp(0, 1)
    out = out.permute(1, 2, 0).numpy()
    return out

# -----------------------
# UI
# -----------------------
st.title("🎨 Pix2Pix Sketch → Color Generator")

uploaded_file = st.file_uploader("Upload Sketch Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Input Sketch", use_container_width=True)

    result = process_image(image)

    st.image(result, caption="Generated Image", use_container_width=True)
