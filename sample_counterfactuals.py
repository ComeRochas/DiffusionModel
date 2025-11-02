import torch
import torch.nn.functional as F
import torchvision
import argparse
from models import UNet, Classifier
from diffusion import GaussianDiffusion
from tqdm import tqdm
import os

def sample_guided(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    model = UNet(in_channels=3, out_channels=3, n_channels=args.n_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    classifier = Classifier(in_channels=3, out_classes=args.num_classes).to(device)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
    classifier.eval()

    diffusion = GaussianDiffusion(model, args.img_size, 3, args.timesteps)

    # Guided p_sample function
    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * args.guidance_scale

    @torch.no_grad()
    def p_sample_guided_loop(shape, target_class):
        device = next(model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)

        y = torch.full((b,), target_class, device=device, dtype=torch.long)

        for i in tqdm(reversed(range(0, diffusion.num_timesteps)), desc='guided sampling loop', total=diffusion.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            # Standard p_sample logic from diffusion.py
            betas_t = diffusion.betas[t, None, None, None].to(device)
            sqrt_one_minus_alphas_cumprod_t = diffusion.sqrt_one_minus_alphas_cumprod[t, None, None, None].to(device)
            sqrt_recip_alphas_t = diffusion.sqrt_recip_alphas[t, None, None, None].to(device)

            model_mean = sqrt_recip_alphas_t * (img - betas_t * model(img, t) / sqrt_one_minus_alphas_cumprod_t)

            # Add guidance
            gradient = cond_fn(img, t, y)
            posterior_variance_t = diffusion.posterior_variance[t, None, None, None].to(device)
            model_mean = model_mean + posterior_variance_t * gradient

            if i == 0:
                img = model_mean
            else:
                noise = torch.randn_like(img)
                img = model_mean + torch.sqrt(posterior_variance_t) * noise

        return img

    # Generate and save images
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(args.num_images):
        print(f"Generating image {i+1}/{args.num_images} for class {args.target_class}")
        sample = p_sample_guided_loop((1, 3, args.img_size, args.img_size), args.target_class)
        sample = (sample + 1) / 2 # unnormalize
        save_path = os.path.join(args.save_dir, f"class_{args.target_class}_img_{i}.png")
        torchvision.utils.save_image(sample, save_path)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained UNet model.")
    parser.add_argument('--classifier_path', type=str, required=True, help="Path to the trained classifier.")
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--n_channels', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--target_class', type=int, default=0, help="The target class for guided generation.")
    parser.add_argument('--save_dir', type=str, default='results/counterfactuals')
    args = parser.parse_args()
    sample_guided(args)
