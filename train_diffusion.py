import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from models import UNet
from diffusion import GaussianDiffusion

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = UNet(in_channels=3, out_channels=3, n_channels=args.n_channels).to(device)
    diffusion = GaussianDiffusion(model, args.img_size, 3, args.timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = torch.randint(0, args.timesteps, (images.shape[0],), device=device).long()

            loss = diffusion.train_step(images, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--n_channels', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--save_path', type=str, default='unet.pth')
    args = parser.parse_args()
    train(args)
