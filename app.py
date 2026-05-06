import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import urllib.request
import os
from model import Generator

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Download weights from GitHub Release
# -----------------------
WEIGHT_URL = "https://github.com/Mubashir214/Conditional-GAN/releases/download/v1.0.0/cuhk_best_generator.pth"
WEIGHT_PATH = "best_generator.pth"

if not os.path.exists(WEIGHT_PATH):
    st.info("Downloading model weights...")
    urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_PATH)

# -----------------------
# Load Model (FIXED VERSION)
# -----------------------
@st.cache_resource
def load_model():
    model = Generator().to(device)

    state_dict = torch.load(WEIGHT_PATH, map_location=device)

    # remove DataParallel prefix
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        cleaned_state_dict[new_key] = v

    # safe load
    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(cleaned_state_dict, strict=False)

    model.eval()
    return model

gen = load_model()

# -----------------------
# Image Transform (MATCH TRAINING)
# -----------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# -----------------------
# Inference Function
# -----------------------
def process_image(img):
    img = img.convert("RGB")

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = gen(img).squeeze(0).cpu()

    output = (output * 0.5 + 0.5).clamp(0, 1)
    output = output.permute(1, 2, 0).numpy()

    return output

# -----------------------
# UI
# -----------------------
st.title("🎨 Pix2Pix Sketch → Colorization App")
st.write("Upload a sketch image and generate colored anime output.")

uploaded_file = st.file_uploader("Upload Sketch", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Sketch", use_container_width=True)

    result = process_image(image)

    with col2:
        st.image(result, caption="Generated Color Image", use_container_width=True)

    st.success("Done! 🎉")
