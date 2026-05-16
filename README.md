# 🎨 Doodle-to-Real Image Translation & Colorization using Pix2Pix

## README.md

# 🖌️ Pix2Pix: Sketch-to-Real & Image Colorization using Conditional GANs

This project implements a **Pix2Pix Conditional GAN (cGAN)** for paired image-to-image translation tasks.

The system learns mappings between paired images to perform:

* ✏️ Sketch / Edge → Realistic Face Image
* 🎨 Grayscale / Anime Sketch → Colored Image

The project demonstrates how conditional GANs can generate high-quality realistic outputs while preserving structural information from the input image.

---

# 🚀 Project Objectives

The main goals of this project are:

* Implement Pix2Pix using PyTorch
* Convert sketches into realistic images
* Perform grayscale image colorization
* Preserve image structure using U-Net skip connections
* Improve local texture realism using PatchGAN
* Deploy a real-time web application using Streamlit or Gradio

---

# 🧠 Concepts Covered

* Conditional GANs (cGAN)
* Pix2Pix Architecture
* U-Net Generator
* PatchGAN Discriminator
* Image-to-Image Translation
* Sketch-to-Image Synthesis
* Image Colorization
* Adversarial Training
* Reconstruction Loss (L1 Loss)

---

# 📂 Dataset Used

## 1️⃣ CUHK Face Sketch Dataset (CUFS)

[CUHK Face Sketch Dataset](https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs?utm_source=chatgpt.com)

Used for:

* Sketch → Real Face Translation

---

## 2️⃣ Anime Sketch Colorization Dataset

[Anime Sketch Colorization Dataset](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair?utm_source=chatgpt.com)

Used for:

* Anime Sketch → Colored Anime Image

---

# ⚙️ Environment Setup

## Platform

* Kaggle Notebook

## Hardware

* GPU: Tesla T4 ×2

## Libraries Used

```bash id="dhocpj"
torch
torchvision
numpy
matplotlib
opencv-python
streamlit
Pillow
tqdm
scikit-image
```

Install dependencies:

```bash id="qfjlwm"
pip install torch torchvision matplotlib pillow tqdm opencv-python scikit-image streamlit
```

---

# 🏗️ Model Architecture

# 🎯 Pix2Pix Conditional GAN

Pix2Pix consists of two networks:

1. **Generator (U-Net)**
2. **Discriminator (PatchGAN)**

---

# 1️⃣ Generator — U-Net Architecture

## Features

* Encoder-Decoder structure
* Skip Connections
* Preserves spatial information
* Maintains fine details

## Input

* Sketch / Edge / Grayscale Image

## Output

* Realistic or Colored Image

## Activations

* Encoder: LeakyReLU
* Decoder: ReLU
* Final Layer: Tanh

---

# 2️⃣ Discriminator — PatchGAN

## Features

* Patch-based classification
* Operates on local image patches
* Improves texture quality
* Detects local realism

## Patch Size

* 16 × 16 patches

## Output

* Matrix of probabilities

Instead of classifying the whole image, PatchGAN determines whether each image patch is real or fake.

---

# 📁 Project Structure

```bash id="lz1jrg"
Pix2Pix-Image-Translation/
│
├── notebooks/
│   ├── pix2pix_training.ipynb
│
├── models/
│   ├── generator.py
│   ├── discriminator.py
│
├── datasets/
│   ├── sketches/
│   ├── real_images/
│
├── checkpoints/
│   ├── generator/
│   ├── discriminator/
│
├── outputs/
│   ├── generated_samples/
│   ├── training_logs/
│
├── app/
│   ├── streamlit_app.py
│
├── requirements.txt
├── README.md
```

---

# 📊 Data Preparation

The preprocessing pipeline includes:

1. Load paired images
2. Resize images to 256 × 256
3. Convert images to tensors
4. Normalize images to range [-1, 1]
5. Create PyTorch DataLoader

Example normalization:

```python id="stzhci"
transforms.Normalize((0.5,), (0.5,))
```

---

# 🔄 Forward Pass Workflow

## Step-by-Step Process

1. Input sketch/grayscale image into Generator
2. Generate fake realistic/colored image
3. Pass real pair into Discriminator
4. Pass fake pair into Discriminator
5. Compute GAN Loss
6. Compute L1 Reconstruction Loss
7. Update Generator and Discriminator alternately

---

# 📉 Loss Functions

# A. Adversarial Loss

Encourages realistic image generation.

```text
GAN Loss = Real vs Fake Classification
```

---

# B. L1 Reconstruction Loss

Encourages generated output to resemble the ground truth image.

```text
L1 Loss = |Generated Image - Real Image|
```

---

# C. Total Generator Loss

```text
Total Loss = GAN Loss + λ × L1 Loss
```

---

# ⚡ Training Configuration

| Parameter       | Value        |
| --------------- | ------------ |
| Optimizer       | Adam         |
| Learning Rate   | 0.0002       |
| Betas           | (0.5, 0.999) |
| Batch Size      | 16–32        |
| Image Size      | 256×256      |
| Mixed Precision | Enabled      |

---

# 🚀 Training Strategy

* Train Generator and Discriminator alternately
* Use paired supervised datasets
* Monitor:

  * Generator Loss
  * Discriminator Loss
  * L1 Loss

---

# ⚙️ Optimization Techniques

To efficiently train on Kaggle T4 GPUs:

* Mixed Precision Training (`torch.cuda.amp`)
* Reduced image resolution if needed (128×128)
* Dataset subsets for fast experimentation
* Checkpoint saving every 5–10 epochs
* GPU memory optimization

---

# 📈 Training Logs

The project includes:

* Generator Loss vs Epochs
* Discriminator Loss vs Epochs
* Sample generated outputs during training

---

# 🖼️ Visualization Module

The visualization utility displays:

| Input        | Generated Output | Ground Truth       |
| ------------ | ---------------- | ------------------ |
| Sketch Image | Realistic Face   | Original Face      |
| Anime Sketch | Colored Anime    | Real Colored Image |

At least:

* 5–10 generated samples per model

---

# 📊 Quantitative Evaluation

## 1️⃣ Structural Similarity Index (SSIM)

Measures similarity between generated and target images.

Higher SSIM = Better structure preservation.

---

## 2️⃣ Peak Signal-to-Noise Ratio (PSNR)

Measures reconstruction quality.

Higher PSNR = Better image quality.

---

# 📱 App Deployment

The project includes a Streamlit/Gradio application that:

✅ Accepts sketch or grayscale image input
✅ Generates realistic/colorized output
✅ Displays results in real-time
✅ Allows easy testing of trained models

Run locally:

```bash id="f8hvwl"
streamlit run streamlit_app.py
```

---

# 🔍 Results & Observations

## Pix2Pix Performance

### Advantages

* Preserves spatial structure
* Generates realistic textures
* Produces high-quality outputs
* Learns paired image translation effectively

### Challenges

* Requires paired datasets
* Training can be computationally expensive
* Sensitive to hyperparameter tuning

---

# 🎯 Applications

This system can be used in:

* AI Art Generation
* Anime Colorization
* Face Reconstruction
* Sketch-to-Photo Translation
* Digital Entertainment
* Computer Vision Research

---

# 🔮 Future Improvements

* High-resolution image generation
* Attention-based U-Net
* CycleGAN for unpaired translation
* Diffusion-based image translation
* Multi-domain image translation
* Faster inference optimization

---

# 🎓 Conclusion

This project successfully demonstrates the power of **Pix2Pix Conditional GANs** for image-to-image translation tasks.

Using:

* **U-Net Generator**
* **PatchGAN Discriminator**
* **Adversarial + L1 Loss**

the model learns to generate realistic and structurally accurate outputs from sketches and grayscale inputs.

The system effectively performs:

* Sketch-to-Real translation
* Anime sketch colorization
* Paired image synthesis

with visually convincing results.

---

# 👨‍💻 Author

**Mubashir Siddique**

AI / Deep Learning / Computer Vision Enthusiast

---

# 📜 License

This project is developed for educational and research purposes.
