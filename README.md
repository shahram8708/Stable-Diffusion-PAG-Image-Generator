# Stable Diffusion PAG Image Generator

This project leverages **Stable Diffusion XL** (SDXL) with **Perturbed-Attention Guidance (PAG)** to generate stunning AI-powered visuals. You can create two variations of an image: one using PAG and another without PAG, and compare the results side-by-side.

## Features
- Generate high-quality images with and without PAG.
- Customizable guidance scales, PAG scales, and layers.
- Ability to use custom LoRA models for fine-tuning.
- Randomized or fixed seed for unique and reproducible outputs.
- Clean and user-friendly UI built with Gradio.

## Requirements
Ensure you have the following installed:
- Python 3.8+
- CUDA-enabled GPU for faster inference.

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/shahram8708/stable-diffusion-pag-image-gen.git
cd stable-diffusion-pag-image-gen
pip install -r requirements.txt
