"""
Training script for diffusion model on MNIST dataset.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from models import UNet, Classifier
from diffusion import DiffusionProcess


def train_diffusion_model(
    epochs=50,
    batch_size=128,
    lr=2e-4,
    timesteps=1000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_interval=10,
    checkpoint_dir='checkpoints'
):
    """
    Train the diffusion model on MNIST.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        timesteps: Number of diffusion timesteps
        device: Device to train on
        save_interval: Save checkpoint every N epochs
        checkpoint_dir: Directory to save checkpoints
    """
    print(f"Training on device: {device}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model and diffusion process
    model = UNet(img_channels=1).to(device)
    diffusion = DiffusionProcess(timesteps=timesteps, device=device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            batch_size_current = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size_current,), device=device)
            
            # Compute loss
            loss = diffusion.compute_loss(model, images, t)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            # Generate and save samples
            model.eval()
            with torch.no_grad():
                samples = diffusion.p_sample_loop(
                    model,
                    shape=(16, 1, 28, 28)
                )
                samples = (samples + 1) / 2  # Scale back to [0, 1]
                samples = samples.cpu().numpy()
                
                # Create grid
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(samples[i, 0], cmap='gray')
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f'results/samples_epoch_{epoch+1}.png')
                plt.close()
                print(f"Saved samples: results/samples_epoch_{epoch+1}.png")
    
    # Final save
    final_path = os.path.join(checkpoint_dir, 'diffusion_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"Training complete! Final model saved: {final_path}")


def train_classifier(
    epochs=10,
    batch_size=128,
    lr=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir='checkpoints'
):
    """
    Train a classifier on MNIST for guidance.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
    """
    print(f"Training classifier on device: {device}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize classifier
    classifier = Classifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    
    # Training loop
    print("Starting classifier training...")
    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Evaluate on test set
        classifier.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = classifier(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    # Save final classifier
    classifier_path = os.path.join(checkpoint_dir, 'classifier.pt')
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'test_accuracy': test_acc,
    }, classifier_path)
    print(f"Classifier training complete! Model saved: {classifier_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train diffusion model and classifier')
    parser.add_argument('--mode', type=str, default='both', choices=['diffusion', 'classifier', 'both'],
                        help='What to train: diffusion model, classifier, or both')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for diffusion training')
    parser.add_argument('--classifier-epochs', type=int, default=10,
                        help='Number of epochs for classifier training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate for diffusion model')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    
    args = parser.parse_args()
    
    if args.mode in ['classifier', 'both']:
        train_classifier(
            epochs=args.classifier_epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            device=args.device
        )
    
    if args.mode in ['diffusion', 'both']:
        train_diffusion_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            timesteps=args.timesteps,
            device=args.device
        )
