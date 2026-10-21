# # Tutorial 15: Generative AI beyond text
# 
# **Prerequisite:** lecture 15. **Compute:** CPU; standard library and NumPy.
# 
# Predict the result before each code cell. Run cells in order from a fresh kernel. Tutorials 01–16 run without coursekit. Tutorials 17–32 use the extracted full course pack linked on the tutorial page. Reference outputs are labeled below; keep your own actual results.

import random
from pathlib import Path
import sys, json, math, time, random
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
COURSE = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RUNS = COURSE / 'runs'
RUNS.mkdir(exist_ok=True)
random.seed(7)
print('Outputs directory:', RUNS)


# ### Pause before the next step
# For x0=2, noise=-1, alpha_bar=.64, compute the noisy scalar by hand.

# Diffusion has a forward noising process and a learned reverse task. Here the clean signal and noise are known, so we can verify the forward equation exactly. Reconstructing with known noise is an algebra check, not a trained denoiser.

import numpy as np
rng = np.random.default_rng(7)
x0 = np.zeros((8, 8))
x0[2:6, 3:5] = 1
noise = rng.normal(size=x0.shape)
for alpha_bar in [0.99, 0.64, 0.1]:
    xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise
    recovered = (xt - np.sqrt(1 - alpha_bar) * noise) / np.sqrt(alpha_bar)
    assert np.allclose(recovered, x0)
    print('alpha_bar', alpha_bar, 'noisy center', np.round(xt[2:6, 2:6], 2))


# ### A schedule and imperfect noise predictions
# Compare the known-noise identity with an imperfect predictor on a stated schedule.

import numpy as np
betas = np.linspace(0.0001, 0.02, 1000)
alpha_bars = np.cumprod(1 - betas)
assert 0 < alpha_bars[-1] < alpha_bars[0] < 1
x0, epsilon, step = (2.0, -1.0, 499)
alpha_bar = alpha_bars[step]
xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * epsilon
epsilon_hat = -0.8
noise_loss = (epsilon_hat - epsilon) ** 2
x0_hat = (xt - np.sqrt(1 - alpha_bar) * epsilon_hat) / np.sqrt(alpha_bar)
assert not np.isclose(x0_hat, x0) and np.isclose(noise_loss, 0.04)
print('Noise-prediction squared error:', round(noise_loss, 4))
print('Schedule endpoints:', alpha_bars[0], alpha_bars[-1], 'Imperfect reconstruction:', x0_hat)


# ### Independent experiment
# Write what inputs a learned noise predictor receives and what target it sees. Explain why using the exact noise for reconstruction here is different from generating an unseen image. Contrast an autoregressive image-token sequence.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# alpha_bar 0.99 noisy center [[-0.19  0.87  0.81 -0.02]
#  [-0.25  0.94  0.99  0.01]
#  [ 0.11  0.91  0.99  0.09]
#  [-0.12  1.    1.13 -0.15]]
# alpha_bar 0.64 noisy center [[-1.14  0.03 -0.31 -0.14]
#  [-1.51  0.48  0.77  0.07]
#  [ 0.64  0.32  0.78  0.53]
#  [-0.74  0.85  1.62 -0.93]]
# alpha_bar 0.1 noisy center [[-1.8  -0.91 -1.43 -0.22]
#  [-2.39 -0.19  0.27  0.11]
#  [ 1.01 -0.45  0.29  0.84]
#  [-1.16  0.39  1.61 -1.47]]
# Noise-prediction squared error: 0.04
# Schedule endpoints: 0.9999 4.035829765375676e-05 Imperfect reconstruction: 1.315172676191192
# ```
