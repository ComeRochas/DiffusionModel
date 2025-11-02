# Diffusion-based Generative Model with Causal Guidance

This repository contains an implementation of a diffusion-based generative model with causal guidance. The model is trained on the CIFAR-10 dataset and can be used to generate counterfactual images.

## Project Structure

- `models.py`: Contains the U-Net and classifier architectures.
- `diffusion.py`: Implements the forward and reverse diffusion processes.
- `train_diffusion.py`: Script for training the diffusion model.
- `train_classifier.py`: Script for training the classifier model.
- `sample_counterfactuals.py`: Script for generating counterfactual images with causal guidance.
- `requirements.txt`: Lists the project dependencies.
- `results/`: Directory to save generated images.

## Dependencies

The project dependencies are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Train the Diffusion Model

First, you need to train the U-Net model. You can do this by running the `train_diffusion.py` script:

```bash
python train_diffusion.py --epochs 100 --batch_size 128 --lr 1e-4
```

This will train the model for 100 epochs and save the trained weights to `unet.pth`.

### 2. Train the Classifier

Next, you need to train the classifier model for the causal guidance. You can do this by running the `train_classifier.py` script:

```bash
python train_classifier.py --epochs 20 --batch_size 128 --lr 1e-3
```
This will train the classifier for 20 epochs and save the trained weights to `classifier.pth`.


### 3. Generate Counterfactual Images

Once you have the trained diffusion model and the classifier, you can generate counterfactual images using the `sample_counterfactuals.py` script:

```bash
python sample_counterfactuals.py \
    --model_path unet.pth \
    --classifier_path classifier.pth \
    --target_class 3 \
    --num_images 5 \
    --guidance_scale 1.5
```

This will generate 5 images of class 3 (cat in CIFAR-10) and save them to the `results/counterfactuals` directory. The `guidance_scale` parameter controls the strength of the causal guidance.

## To-Do / Future Work
- [ ] Experiment with different datasets.
- [ ] Implement more advanced guidance techniques.
- [ ] Add support for distributed training.
