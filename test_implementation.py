import torch
import unittest
from models import UNet, Classifier
from diffusion import GaussianDiffusion

class TestImplementation(unittest.TestCase):

    def test_models_forward_pass(self):
        device = torch.device("cpu")

        # Test UNet
        unet = UNet(in_channels=3, out_channels=3, n_channels=32).to(device)
        x_unet = torch.randn(2, 3, 32, 32).to(device)
        t_unet = torch.randint(0, 1000, (2,), device=device).long()
        out_unet = unet(x_unet, t_unet)
        self.assertEqual(out_unet.shape, (2, 3, 32, 32))

        # Test Classifier
        classifier = Classifier(in_channels=3, out_classes=10, n_channels=32).to(device)
        x_classifier = torch.randn(2, 3, 32, 32).to(device)
        out_classifier = classifier(x_classifier)
        self.assertEqual(out_classifier.shape, (2, 10))

    def test_diffusion_q_sample(self):
        device = torch.device("cpu")
        model = UNet(in_channels=3, out_channels=3, n_channels=32).to(device)
        diffusion = GaussianDiffusion(model, 32, 3, num_timesteps=100)

        x_start = torch.randn(2, 3, 32, 32).to(device)
        t = torch.randint(0, 100, (2,), device=device).long()

        x_noisy = diffusion.q_sample(x_start, t)
        self.assertEqual(x_noisy.shape, x_start.shape)

    def test_diffusion_train_step(self):
        device = torch.device("cpu")
        model = UNet(in_channels=3, out_channels=3, n_channels=32).to(device)
        diffusion = GaussianDiffusion(model, 32, 3, num_timesteps=100)

        x_start = torch.randn(2, 3, 32, 32).to(device)
        t = torch.randint(0, 100, (2,), device=device).long()

        loss = diffusion.train_step(x_start, t)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0) # loss should be a scalar

if __name__ == '__main__':
    unittest.main()
