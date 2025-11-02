# Results Directory

This directory will contain generated images:
- `sample_unconditional.png`: Unconditional samples without guidance
- `sample_guided.png`: Class-conditional samples with classifier guidance (with different coefficients)
- `all_digits_guided.png`: Grid of 10 fixed inputs producing, for each digit, a different output via guided denoising

Run `sample_counterfactuals.py` after training to generate these images.


Discussion : 
Guided samples were qualitatively less accurate than unconditional ones. This is expected because the classifier used for guidance was trained only on clean data, while the diffusion model operates over noisy intermediate states.
As a result, classifier gradients become unreliable at high noise levels, perturbing the reverse diffusion trajectory. 
In practice, one could train a noise-aware classifier to obtain sharper conditional samples.
