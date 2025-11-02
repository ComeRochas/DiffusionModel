# Diffusion Model with Causal Guidance

A PyTorch implementation of a diffusion-based generative model with classifier-guided sampling for counterfactual generation. This project intends to demonstrate controllable image generation on MNIST by intervening on class labels using causal guidance.

## Overview

This repository implements:
- **Denoising Diffusion Probabilistic Models (DDPM)** for image generation
- **U-Net architecture** for the denoising process
- **Classifier-guided diffusion** for controllable generation
- **Counterfactual sampling** through causal intervention on class labels

## Project Structure

```
DiffusionModel/
├── models.py                    # U-Net and Classifier architectures
├── diffusion.py                 # Forward & reverse diffusion process
├── train_diffusion.py           # Training loop for model and classifier
├── sample_counterfactuals.py    # Counterfactual generation with guidance
├── requirements.txt             # Python dependencies
├── results/                     # Generated samples (created during sampling)
│   ├── sample_unconditional.png
│   ├── sample_guided.png
│   └── all_digits_guided.png
├── training_results/            # Generated samples every 10 epochs of training
├── checkpoints/                 # Saved model checkpoints (created during training)
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ComeRochas/DiffusionModel.git
cd DiffusionModel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

Train both the diffusion model and classifier:

```bash
python train_diffusion.py --mode both --epochs 50 --classifier-epochs 10
```

Options:
- `--mode`: What to train (`diffusion`, `classifier`, or `both`)
- `--epochs`: Number of epochs for diffusion model (default: 50)
- `--classifier-epochs`: Number of epochs for classifier (default: 10)
- `--batch-size`: Batch size for training (default: 128)
- `--lr`: Learning rate for diffusion model (default: 2e-4)
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--device`: Device to train on (`cuda` or `cpu`)

The training script will:
- Download MNIST dataset automatically
- Train the classifier for digit recognition
- Train the diffusion model for image generation
- Save checkpoints in `checkpoints/` directory
- Generate sample images during training in `results/` directory

### 2. Generating Counterfactual Samples

Generate counterfactual samples using the trained models:

```bash
python sample_counterfactuals.py \
    --diffusion-checkpoint checkpoints/diffusion_final.pt \
    --classifier-checkpoint checkpoints/classifier.pt \
    --guidance-scale 5.0 \
    --source-class 8 
```

Options:
- `--diffusion-checkpoint`: Path to trained diffusion model
- `--classifier-checkpoint`: Path to trained classifier
- `--guidance-scale`: Strength of classifier guidance (higher = stronger, default: 5.0)
- `--source-class`: Source class for counterfactual generation (0-9)
- `--target-class`: Target class for counterfactual generation (0-9)
- `--n-samples`: Number of samples to generate (default: 16)
- `--output-dir`: Directory to save results (default: `results/`)

This will generate:
1. **Unconditional samples**: Random digit generation without guidance
2. **Guided samples**: Digits conditioned on a specific class
3. **All digits conditioning**: Side-by-side comparison of denoising the same samples 10 times, by conditioned on the digit classes

## How It Works

### Diffusion Process

1. **Forward Process (Noising)**: Gradually adds Gaussian noise to images over T timesteps until they become pure noise
2. **Reverse Process (Denoising)**: Learns to remove noise step-by-step to generate new images

### Classifier Guidance

The model uses classifier guidance to control generation:
- A classifier predicts the class label from noisy images
- During generation, gradients from the classifier guide the denoising process toward the target class
- This enables **causal intervention**: changing what class we want to generate

- In practice, the classifier was trained on original MNIST images only. Therefore, when confronted to noised images, its gradients are poorly indicative and classifier guidance happens to lack precision. Training a classifier on progressively noised images could potentially solve this issue.

### Counterfactual Generation

Counterfactuals answer: "What would this image look like if it belonged to a different class?"

By intervening on the class label during generation:
- Generate an image conditioned on class 3 → Get a "3" digit
- Generate an image conditioned on class 8 → Get an "8" digit
- Compare them to understand the causal effect of the class label

## Results

After training and sampling, you'll find in `results/`:

- `sample_unconditional.png`: Random samples without guidance
- `sample_guided.png`: Samples guided toward a specific class
- `all_digits_guided.png`: 10x10 grid of all 10 digit classes conditioning denoising for 10 fixed inputs


## Technical Details

### Model Architecture

- **U-Net**: Encoder-decoder with skip connections, time embeddings, and group normalization
- **Classifier**: CNN with 3 convolutional layers and 2 fully connected layers

### Training

- **Diffusion Loss**: Mean Squared Error between predicted and actual noise
- **Classifier Loss**: Cross-entropy for digit classification
- **Optimizer**: Adam with learning rate 2e-4 (diffusion) and 1e-3 (classifier)
- **Timesteps**: 1000 diffusion steps with linear beta schedule

### Guidance

The guidance mechanism modifies the predicted noise using:
```
ε_guided = ε_predicted - scale * ∇_x log p(y|x)
```
where `scale` controls guidance strength and the gradient comes from the classifier.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended) or CPU

## Citation

This implementation is based on:
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Diffusion Models Beat GANs on Image Synthesis (Dhariwal & Nichol, 2021)](https://arxiv.org/abs/2105.05233)
