"""
Quick validation test to ensure all components work correctly.
This test doesn't train the full model but validates that all APIs work.
"""
import torch
from models import UNet, Classifier
from diffusion import DiffusionProcess


def test_models():
    """Test that models can be instantiated and run forward passes."""
    print("=" * 60)
    print("Testing Models...")
    print("=" * 60)
    
    # Test UNet
    print("\n1. Testing UNet...")
    model = UNet(img_channels=1)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   ✓ UNet created with {param_count:,} parameters")
    
    # Forward pass
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))
    noise_pred = model(x, t)
    assert noise_pred.shape == x.shape, "UNet output shape mismatch"
    print(f"   ✓ UNet forward pass: {x.shape} -> {noise_pred.shape}")
    
    # Test Classifier
    print("\n2. Testing Classifier...")
    classifier = Classifier(num_classes=10)
    param_count = sum(p.numel() for p in classifier.parameters())
    print(f"   ✓ Classifier created with {param_count:,} parameters")
    
    # Forward pass
    logits = classifier(x)
    assert logits.shape == (4, 10), "Classifier output shape mismatch"
    print(f"   ✓ Classifier forward pass: {x.shape} -> {logits.shape}")
    
    print("\n✅ All model tests passed!")
    return model, classifier


def test_diffusion(model):
    """Test diffusion process."""
    print("\n" + "=" * 60)
    print("Testing Diffusion Process...")
    print("=" * 60)
    
    diffusion = DiffusionProcess(timesteps=100, device='cpu')
    print("   ✓ DiffusionProcess created")
    
    # Test forward process (adding noise)
    print("\n1. Testing forward diffusion (noising)...")
    x_0 = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 100, (4,))
    x_t = diffusion.q_sample(x_0, t)
    assert x_t.shape == x_0.shape, "Forward diffusion shape mismatch"
    print(f"   ✓ Added noise: x_0 {x_0.shape} -> x_t {x_t.shape}")
    
    # Test loss computation
    print("\n2. Testing loss computation...")
    loss = diffusion.compute_loss(model, x_0, t)
    print(f"   ✓ Loss computed: {loss.item():.4f}")
    
    # Test reverse process (denoising)
    print("\n3. Testing reverse diffusion (denoising)...")
    x_t = torch.randn(2, 1, 28, 28)
    x_t_minus_1 = diffusion.p_sample(model, x_t, t=50)
    assert x_t_minus_1.shape == x_t.shape, "Reverse step shape mismatch"
    print(f"   ✓ Denoising step: x_t {x_t.shape} -> x_(t-1) {x_t_minus_1.shape}")
    
    # Test full sampling loop (with fewer timesteps for speed)
    print("\n4. Testing full sampling loop...")
    samples = diffusion.p_sample_loop(model, shape=(2, 1, 28, 28))
    assert samples.shape == (2, 1, 28, 28), "Sampling output shape mismatch"
    print(f"   ✓ Generated samples: {samples.shape}")
    
    print("\n✅ All diffusion tests passed!")
    return diffusion


def test_guided_sampling(model, classifier, diffusion):
    """Test classifier-guided sampling."""
    print("\n" + "=" * 60)
    print("Testing Classifier-Guided Sampling...")
    print("=" * 60)
    
    # Test guided sampling with a single class
    print("\n1. Testing guided sampling for a single class...")
    target_class = 7
    samples = diffusion.p_sample_loop(
        model,
        shape=(2, 1, 28, 28),
        classifier=classifier,
        guidance_scale=3.0,
        target_class=target_class
    )
    assert samples.shape == (2, 1, 28, 28), "Guided sampling shape mismatch"
    print(f"   ✓ Generated guided samples for class {target_class}: {samples.shape}")
    
    # Test guided sampling with different classes
    print("\n2. Testing guided sampling for multiple classes...")
    target_classes = [0, 1]
    samples = diffusion.p_sample_loop(
        model,
        shape=(2, 1, 28, 28),
        classifier=classifier,
        guidance_scale=3.0,
        target_class=target_classes
    )
    assert samples.shape == (2, 1, 28, 28), "Multi-class guided sampling shape mismatch"
    print(f"   ✓ Generated guided samples for classes {target_classes}: {samples.shape}")
    
    print("\n✅ All guided sampling tests passed!")


def test_counterfactual_concept():
    """Test the counterfactual concept explanation."""
    print("\n" + "=" * 60)
    print("Demonstrating Counterfactual Concept...")
    print("=" * 60)
    
    print("\nCounterfactual generation allows us to answer:")
    print("'What would this image look like if it belonged to a different class?'")
    print()
    print("Example workflow:")
    print("  1. Generate image conditioned on class 3 → Get digit '3'")
    print("  2. Generate image conditioned on class 8 → Get digit '8'")
    print("  3. Compare to understand causal effect of class label")
    print()
    print("This demonstrates CAUSAL INTERVENTION on the class label!")
    print()
    print("To see this in action:")
    print("  1. Train the models: python train_diffusion.py --mode both --epochs 50")
    print("  2. Generate counterfactuals: python sample_counterfactuals.py")
    print("  3. View results in: results/counterfactual_example.png")
    
    print("\n✅ Counterfactual concept explained!")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("DIFFUSION MODEL VALIDATION TEST")
    print("=" * 60)
    print()
    print("This script validates that all components work correctly.")
    print("It does NOT train models, only tests the API and architecture.")
    print()
    
    # Run tests
    model, classifier = test_models()
    diffusion = test_diffusion(model)
    test_guided_sampling(model, classifier, diffusion)
    test_counterfactual_concept()
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\n✅ All components are working correctly!")
    print("\nNext steps:")
    print("  1. Train models: python train_diffusion.py --mode both")
    print("  2. Generate samples: python sample_counterfactuals.py")
    print("\nSee README.md for detailed usage instructions.")
    print()


if __name__ == '__main__':
    main()
