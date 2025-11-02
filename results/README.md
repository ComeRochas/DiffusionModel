# Results Directory

This directory will contain generated images:
- `sample_unconditional.png`: Unconditional samples without guidance
- `sample_guided.png`: Class-conditional samples with classifier guidance (with different coefficients)
- `all_digits_guided.png`: Grid of 10 fixed inputs producing, for each digit, a different output via guided denoising

Run `sample_counterfactuals.py` after training to generate these images.
