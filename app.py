import streamlit as st
import torch
import os
from model import Generator
from utils import generate_images

# ---------------- UI ----------------
st.set_page_config(page_title="GAN Generator", layout="wide")
st.title("🎨 GAN Image Generator")
st.markdown("Compare DCGAN vs WGAN-GP")

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.success(f"Using device: {device}")

# ---------------- File Paths ----------------
DCGAN_PATH = "G_dcgan_final.pth"
WGAN_PATH = "G_wgan_final.pth"

# ---------------- File Check ----------------
st.sidebar.subheader("📦 Model Status")

if os.path.exists(DCGAN_PATH):
    st.sidebar.success("✅ DCGAN weight file found")
else:
    st.sidebar.error("❌ DCGAN weight file missing")

if os.path.exists(WGAN_PATH):
    st.sidebar.success("✅ WGAN weight file found")
else:
    st.sidebar.error("❌ WGAN weight file missing")

# Stop if missing
if not os.path.exists(DCGAN_PATH) or not os.path.exists(WGAN_PATH):
    st.error("❌ One or more model files are missing. Please upload them.")
    st.stop()

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    z_dim = 100

    # DCGAN
    G_dcgan = Generator(z_dim).to(device)
    try:
        state_dcgan = torch.load(DCGAN_PATH, map_location=device)
        G_dcgan.load_state_dict(state_dcgan)
        G_dcgan.eval()
        dcgan_loaded = True
    except Exception as e:
        st.sidebar.error(f"❌ DCGAN load failed: {e}")
        dcgan_loaded = False

    # WGAN
    G_wgan = Generator(z_dim).to(device)
    try:
        state_wgan = torch.load(WGAN_PATH, map_location=device)
        G_wgan.load_state_dict(state_wgan)
        G_wgan.eval()
        wgan_loaded = True
    except Exception as e:
        st.sidebar.error(f"❌ WGAN load failed: {e}")
        wgan_loaded = False

    return G_dcgan, G_wgan, dcgan_loaded, wgan_loaded

G_dcgan, G_wgan, dcgan_loaded, wgan_loaded = load_models()

# ---------------- Load Status Messages ----------------
st.sidebar.subheader("🧠 Model Loading Status")

if dcgan_loaded:
    st.sidebar.success("✅ DCGAN model loaded successfully")
else:
    st.sidebar.error("❌ DCGAN model NOT loaded")

if wgan_loaded:
    st.sidebar.success("✅ WGAN-GP model loaded successfully")
else:
    st.sidebar.error("❌ WGAN-GP model NOT loaded")

# ---------------- Controls ----------------
model_choice = st.sidebar.radio("Choose Model", ["DCGAN", "WGAN-GP"])
num_images = st.sidebar.slider("Number of Images", 4, 64, 16)

# ---------------- Generate ----------------
if st.button("🚀 Generate Images"):

    if model_choice == "DCGAN" and dcgan_loaded:
        img = generate_images(G_dcgan, device=device, n=num_images)
        st.image(img, caption="DCGAN Output", use_container_width=True)

    elif model_choice == "WGAN-GP" and wgan_loaded:
        img = generate_images(G_wgan, device=device, n=num_images)
        st.image(img, caption="WGAN-GP Output", use_container_width=True)

    else:
        st.error("❌ Selected model is not loaded properly")

# ---------------- Compare ----------------
st.markdown("---")
st.subheader("🔍 Compare Models")

if st.button("Compare DCGAN vs WGAN-GP"):

    if dcgan_loaded and wgan_loaded:
        col1, col2 = st.columns(2)

        img1 = generate_images(G_dcgan, device=device)
        img2 = generate_images(G_wgan, device=device)

        with col1:
            st.image(img1, caption="DCGAN", use_container_width=True)

        with col2:
            st.image(img2, caption="WGAN-GP", use_container_width=True)
    else:
        st.error("❌ Cannot compare — one or both models failed to load")
