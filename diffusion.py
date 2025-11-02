"""
Diffusion process implementation: forward noising and reverse denoising.
Implements DDPM (Denoising Diffusion Probabilistic Models).
"""
import torch
import torch.nn.functional as F
import numpy as np


class DiffusionProcess:
    """
    Implements the forward and reverse diffusion process.
    
    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting value of beta schedule
        beta_end: Ending value of beta schedule
        device: Device to run on (cuda/cpu)
    """
    
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: add noise to x_0 to get x_t.
        
        Args:
            x_0: Original images [batch_size, channels, height, width]
            t: Timesteps [batch_size]
            noise: Optional pre-generated noise
            
        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        # q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, classifier=None, guidance_scale=0.0, target_class=None):
        """
        Single step of reverse diffusion process: denoise x_t to get x_{t-1}.
        Optionally uses classifier guidance.
        
        Args:
            model: Denoising model (U-Net)
            x_t: Noisy images at timestep t
            t: Current timestep
            classifier: Optional classifier for guidance
            guidance_scale: Strength of classifier guidance
            target_class: Target class for guided generation
            
        Returns:
            Denoised images at timestep t-1
        """
        batch_size = x_t.shape[0]
        
        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        
        # Predict noise
        if classifier is not None and guidance_scale > 0 and target_class is not None:
            # Enable gradients for guidance
            x_t = x_t.detach().requires_grad_(True)
            
            predicted_noise = model(x_t, t_tensor)
            
            # Get classifier predictions
            logits = classifier(x_t)
            
            # Compute gradient of log probability w.r.t. x_t
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[range(batch_size), target_class]
            
            # Gradient of log p(y|x_t) w.r.t. x_t
            grad = torch.autograd.grad(selected_log_probs.sum(), x_t)[0]
            
            # Modify noise prediction by gradient
            predicted_noise = predicted_noise - guidance_scale * grad * self.sqrt_one_minus_alphas_cumprod[t]
            
            # Detach x_t from computation graph
            x_t = x_t.detach()
        else:
            predicted_noise = model(x_t, t_tensor)
        
        # Extract values for current timestep
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Predict x_0 from x_t and noise
        pred_x_0 = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x_0 = torch.clamp(pred_x_0, -1, 1)
        
        # Compute mean of p(x_{t-1} | x_t)
        if t > 0:
            # Mean = (sqrt(alpha_t) * (1 - alpha_cumprod_{t-1}) * x_t + 
            #         sqrt(alpha_cumprod_{t-1}) * beta_t * pred_x_0) / (1 - alpha_cumprod_t)
            alpha_cumprod_prev = self.alphas_cumprod_prev[t]
            
            pred_mean = (
                torch.sqrt(alpha_cumprod_prev) * beta_t * pred_x_0 / (1.0 - alpha_cumprod_t) +
                torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev) * x_t / (1.0 - alpha_cumprod_t)
            )
            
            # Add noise
            noise = torch.randn_like(x_t)
            posterior_variance_t = self.posterior_variance[t]
            x_prev = pred_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            # For t=0, just return the predicted x_0
            x_prev = pred_x_0
            
        return x_prev
    
    def p_sample_loop(self, model, shape, classifier=None, guidance_scale=0.0, target_class=None, x_start=None):
        """
        Complete reverse diffusion loop to generate samples from noise.
        
        Args:
            model: Denoising model (U-Net)
            shape: Shape of images to generate [batch_size, channels, height, width]
            classifier: Optional classifier for guidance
            guidance_scale: Strength of classifier guidance
            target_class: Target class for guided generation (single class or list)
            
        Returns:
            Generated images
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=self.device) if x_start is None else x_start.clone()
        
        # Convert target_class to tensor if needed
        if target_class is not None:
            if isinstance(target_class, int):
                target_class = torch.full((batch_size,), target_class, device=self.device, dtype=torch.long)
            elif isinstance(target_class, (list, tuple)):
                target_class = torch.tensor(target_class, device=self.device, dtype=torch.long)
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(
                model, x, t, 
                classifier=classifier, 
                guidance_scale=guidance_scale,
                target_class=target_class
            )
            
        return x
    
    def compute_loss(self, model, x_0, t):
        """
        Compute the denoising loss for training.
        
        Args:
            model: Denoising model (U-Net)
            x_0: Original images
            t: Timesteps
            
        Returns:
            MSE loss between predicted and actual noise
        """
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Add noise to images
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
