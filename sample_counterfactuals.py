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
    # If gradients were enabled during generation, detach before returning so callers
    # can convert to numpy without needing to call detach themselves.
    return samples.detach()


def sample_guided(model, classifier, diffusion, target_class, guidance_scales=5.0, 
                  n_samples=16, device='cuda'):
    """
    Generate class-conditional samples with classifier guidance.
    
    Args:
        model: Diffusion model
        classifier: Classifier for guidance
        diffusion: DiffusionProcess instance
        target_class: Target class to generate (0-9 for MNIST)
        guidance_scales: Either a single guidance scale (float) or a list/tuple of
            guidance scales. If a list is provided, the function will generate
            `n_samples` images for each guidance scale and return a list of
            result tensors (one per guidance scale). If a single float is
            provided, a single tensor is returned (backwards compatible).
        n_samples: Number of samples to generate per guidance scale
        device: Device to generate on
        
    Returns:
        If `guidance_scales` is a scalar: a tensor of shape (n_samples, 1, 28, 28).
        If `guidance_scales` is a list: a list of tensors, each of shape
        (n_samples, 1, 28, 28), corresponding to each guidance scale.
    """
    single_input = not isinstance(guidance_scales, (list, tuple))
    scales = [guidance_scales] if single_input else list(guidance_scales)

    results = []
    for gs in scales:
        print(f"Generating guided samples for class {target_class} with guidance scale {gs}...")
        # Only enable gradients when guidance is requested; keep no_grad for unconditional speed
        with torch.set_grad_enabled(gs > 0 and classifier is not None):
            samples = diffusion.p_sample_loop(
                model,
                shape=(n_samples, 1, 28, 28),
                classifier=classifier,
                guidance_scale=gs,
                target_class=target_class
            )

        # Scale to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1).detach()
        results.append(samples)

    # Backwards compatible: return a single tensor when the user provided a scalar
    return results[0] if single_input else results



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
    # Ensure we operate on a detached CPU numpy array (works whether or not
    # the tensor currently requires grad).
    samples = samples.detach().cpu().numpy()
    
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


def visualize_guided_samples(guided_samples, guidance_scales, target_class, output_dir):
        # guided_samples is a list (one tensor per guidance scale). Create a figure
    # with one row per scale and n_samples columns to compare effects of guidance.
    n_rows = len(guided_samples)
    n_cols = guided_samples[0].shape[0]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))

    # Normalize axes shape for consistent indexing
    axes = np.atleast_2d(axes)

    for i, samples in enumerate(guided_samples):
        samples_np = samples.detach().cpu().numpy()
        for j in range(n_cols):
            ax = axes[i, j]
            ax.imshow(samples_np[j, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

        # Place the guidance scale text to the left of the row using figure coordinates
        pos = axes[i, 0].get_position()  # BBox in figure coordinates
        fig.text(pos.x0 - 0.15, pos.y0 + pos.height / 2,
                 f'{guidance_scales[i]}', fontsize=10,
                 ha='right', va='center')

    plt.suptitle(f'Guided Samples (Class {target_class}) with different Guidance Scales', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'sample_guided.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def visualize_all_digits(classifier, diffusion, model, device, guidance_scale=5.0, output_dir='results'):
    all_digit_samples = []
    # Pre-generate 10 random initial images. Each digit will use the same set of 10 inputs, producing 10 outputs per digit (100 total).
    x_inits = torch.randn((10, 1, 28, 28), device=device)
    n_per_digit = x_inits.shape[0]
    # If guidance is required we need gradients enabled during the p_sample_loop
    with torch.set_grad_enabled(guidance_scale > 0 and classifier is not None):
        for digit in range(10):
            # Use the same 10 inputs for every digit
            samples = diffusion.p_sample_loop(
                model,
                shape=(n_per_digit, 1, 28, 28),
                classifier=classifier,
                guidance_scale=guidance_scale,
                target_class=digit,
                x_start=x_inits
            )

            # Scale and clamp, then detach to avoid requiring grad downstream
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1).detach()
            all_digit_samples.append(samples)

    all_digit_samples = torch.cat(all_digit_samples, dim=0)
    
    # Visualize all digits in a 10x10 grid
    # Detach before converting to numpy in case tensors require gradients
    samples_np = all_digit_samples.detach().cpu().numpy()
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
    plt.savefig(os.path.join(output_dir, 'all_digits_guided.png'), 
                dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'all_digits_guided.png')}")
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
    
    if False:
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
        guidance_scales = [0.5, 2.0, 5.0, 10.0]
        guided_samples = sample_guided(
            model, classifier, diffusion,
            target_class=args.source_class,
            guidance_scales=guidance_scales,
            n_samples=4,
            device=args.device
        )

        visualize_guided_samples(
            guided_samples,
            guidance_scales,
            args.source_class,
            args.output_dir
        )

    # 3. Generate samples for all digits
    print("\n=== Generating All Digits ===")
    visualize_all_digits(
        classifier,
        diffusion,
        model,
        device=args.device,
        guidance_scale=args.guidance_scale,
        output_dir=args.output_dir
    )

    print("\n✓ Counterfactual generation complete!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
