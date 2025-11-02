"""
Sample counterfactual images using classifier-guided diffusion.
Demonstrates causal intervention by changing the target class during generation.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from models import UNet, Classifier
from diffusion import DiffusionProcess


def load_models(diffusion_checkpoint, classifier_checkpoint, device):
    """
    Load pre-trained diffusion model and classifier.
    
    Args:
        diffusion_checkpoint: Path to diffusion model checkpoint
        classifier_checkpoint: Path to classifier checkpoint
        device: Device to load models on
        
    Returns:
        Tuple of (model, classifier, diffusion_process)
    """
    # Load diffusion model
    model = UNet(img_channels=1).to(device)
    checkpoint = torch.load(diffusion_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load classifier
    classifier = Classifier(num_classes=10).to(device)
    checkpoint = torch.load(classifier_checkpoint, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    # Initialize diffusion process
    diffusion = DiffusionProcess(timesteps=1000, device=device)
    
    return model, classifier, diffusion


def sample_unconditional(model, diffusion, n_samples=16, device='cuda'):
    """
    Generate unconditional samples (no guidance).
    
    Args:
        model: Diffusion model
        diffusion: DiffusionProcess instance
        n_samples: Number of samples to generate
        device: Device to generate on
        
    Returns:
        Generated samples
    """
    print("Generating unconditional samples...")
    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model,
            shape=(n_samples, 1, 28, 28)
        )
    
    # Scale to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    return samples


def sample_guided(model, classifier, diffusion, target_class, guidance_scale=5.0, 
                  n_samples=16, device='cuda'):
    """
    Generate class-conditional samples with classifier guidance.
    
    Args:
        model: Diffusion model
        classifier: Classifier for guidance
        diffusion: DiffusionProcess instance
        target_class: Target class to generate (0-9 for MNIST)
        guidance_scale: Strength of classifier guidance
        n_samples: Number of samples to generate
        device: Device to generate on
        
    Returns:
        Generated samples
    """
    print(f"Generating guided samples for class {target_class} with guidance scale {guidance_scale}...")
    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model,
            shape=(n_samples, 1, 28, 28),
            classifier=classifier,
            guidance_scale=guidance_scale,
            target_class=target_class
        )
    
    # Scale to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    return samples


def generate_counterfactual_comparison(model, classifier, diffusion, 
                                      source_class, target_class, 
                                      guidance_scale=5.0, n_pairs=8, 
                                      device='cuda'):
    """
    Generate counterfactual pairs: what if we intervened to change the class?
    
    Args:
        model: Diffusion model
        classifier: Classifier for guidance
        diffusion: DiffusionProcess instance
        source_class: Original class
        target_class: Target class after intervention
        guidance_scale: Strength of classifier guidance
        n_pairs: Number of pairs to generate
        device: Device to generate on
        
    Returns:
        Tuple of (source_samples, target_samples)
    """
    print(f"Generating counterfactual pairs: {source_class} -> {target_class}")
    
    with torch.no_grad():
        # Generate samples conditioned on source class
        source_samples = diffusion.p_sample_loop(
            model,
            shape=(n_pairs, 1, 28, 28),
            classifier=classifier,
            guidance_scale=guidance_scale,
            target_class=source_class
        )
        
        # Generate samples conditioned on target class (counterfactual)
        target_samples = diffusion.p_sample_loop(
            model,
            shape=(n_pairs, 1, 28, 28),
            classifier=classifier,
            guidance_scale=guidance_scale,
            target_class=target_class
        )
    
    # Scale to [0, 1]
    source_samples = (source_samples + 1) / 2
    target_samples = (target_samples + 1) / 2
    source_samples = torch.clamp(source_samples, 0, 1)
    target_samples = torch.clamp(target_samples, 0, 1)
    
    return source_samples, target_samples


def visualize_samples(samples, title, save_path, nrows=4, ncols=4):
    """
    Visualize and save generated samples.
    
    Args:
        samples: Generated samples tensor
        title: Plot title
        save_path: Path to save figure
        nrows: Number of rows in grid
        ncols: Number of columns in grid
    """
    samples = samples.cpu().numpy()
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(samples):
            ax.imshow(samples[i, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_counterfactual_pairs(source_samples, target_samples, 
                                   source_class, target_class, save_path):
    """
    Visualize counterfactual pairs side by side.
    
    Args:
        source_samples: Samples from source class
        target_samples: Samples from target class (counterfactual)
        source_class: Source class label
        target_class: Target class label
        save_path: Path to save figure
    """
    source_samples = source_samples.cpu().numpy()
    target_samples = target_samples.cpu().numpy()
    
    n_pairs = len(source_samples)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(4, n_pairs*2))
    
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_pairs):
        # Source
        axes[i, 0].imshow(source_samples[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title(f'Class {source_class}', fontsize=12)
        
        # Target (counterfactual)
        axes[i, 1].imshow(target_samples[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title(f'Class {target_class}\n(Counterfactual)', fontsize=12)
    
    plt.suptitle(f'Counterfactual Generation: {source_class} → {target_class}', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate counterfactual samples using classifier-guided diffusion'
    )
    parser.add_argument('--diffusion-checkpoint', type=str, 
                        default='checkpoints/diffusion_final.pt',
                        help='Path to diffusion model checkpoint')
    parser.add_argument('--classifier-checkpoint', type=str,
                        default='checkpoints/classifier.pt',
                        help='Path to classifier checkpoint')
    parser.add_argument('--guidance-scale', type=float, default=5.0,
                        help='Classifier guidance scale (higher = stronger guidance)')
    parser.add_argument('--source-class', type=int, default=3,
                        help='Source class for counterfactual generation')
    parser.add_argument('--target-class', type=int, default=8,
                        help='Target class for counterfactual generation')
    parser.add_argument('--n-samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    model, classifier, diffusion = load_models(
        args.diffusion_checkpoint,
        args.classifier_checkpoint,
        args.device
    )
    
    # 1. Generate unconditional samples
    print("\n=== Unconditional Generation ===")
    unconditional_samples = sample_unconditional(
        model, diffusion, n_samples=args.n_samples, device=args.device
    )
    visualize_samples(
        unconditional_samples,
        'Unconditional Samples',
        os.path.join(args.output_dir, 'sample_unconditional.png')
    )
    
    # 2. Generate guided samples for a specific class
    print("\n=== Guided Generation ===")
    guided_samples = sample_guided(
        model, classifier, diffusion,
        target_class=args.source_class,
        guidance_scale=args.guidance_scale,
        n_samples=args.n_samples,
        device=args.device
    )
    visualize_samples(
        guided_samples,
        f'Guided Samples (Class {args.source_class}, Scale {args.guidance_scale})',
        os.path.join(args.output_dir, 'sample_guided.png')
    )
    
    # 3. Generate counterfactual pairs
    print("\n=== Counterfactual Generation ===")
    source_samples, target_samples = generate_counterfactual_comparison(
        model, classifier, diffusion,
        source_class=args.source_class,
        target_class=args.target_class,
        guidance_scale=args.guidance_scale,
        n_pairs=8,
        device=args.device
    )
    visualize_counterfactual_pairs(
        source_samples, target_samples,
        args.source_class, args.target_class,
        os.path.join(args.output_dir, 'counterfactual_example.png')
    )
    
    # 4. Generate samples for all digits
    print("\n=== Generating All Digits ===")
    all_digit_samples = []
    for digit in range(10):
        samples = sample_guided(
            model, classifier, diffusion,
            target_class=digit,
            guidance_scale=args.guidance_scale,
            n_samples=10,
            device=args.device
        )
        all_digit_samples.append(samples)
    
    all_digit_samples = torch.cat(all_digit_samples, dim=0)
    
    # Visualize all digits in a 10x10 grid
    samples_np = all_digit_samples.cpu().numpy()
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            axes[i, j].imshow(samples_np[idx, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].text(-2, 14, str(i), fontsize=14, ha='right', va='center')
    
    plt.suptitle('All Digits (Classifier-Guided Generation)', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'all_digits_guided.png'), 
                dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(args.output_dir, 'all_digits_guided.png')}")
    plt.close()
    
    print("\n✓ Counterfactual generation complete!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
