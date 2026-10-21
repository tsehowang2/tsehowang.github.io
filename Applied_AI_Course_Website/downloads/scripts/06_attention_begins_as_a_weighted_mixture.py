# # Tutorial 06: Attention begins as a weighted mixture
# 
# **Prerequisite:** lecture 06. **Compute:** CPU; standard library and NumPy.
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
# Calculate the two output coordinates separately. Do not treat the values as probabilities.

# Before queries or keys, attention is a weighted mixture. Think of three information cards whose contents are vectors. The weights decide how much of each card to include. For this exercise, the weights are supplied rather than learned.

import numpy as np
values = np.array([[2.0, 0.0], [0.0, 4.0], [2.0, 2.0]])
weights = np.array([0.5, 0.3, 0.2])
out = weights @ values
print('Each weighted contribution:', weights[:, None] * values, 'sum:', out)
assert np.allclose(out, [1.4, 1.6])
assert np.isclose(weights.sum(), 1)
new = np.array([0.2, 0.5, 0.3])
print('Changed mixture:', new @ values)
assert np.allclose(new @ values, [1.0, 2.6])


# ### Independent experiment
# Hold values fixed and sweep the first weight while sharing the remaining mass equally. Plot or tabulate the output. Explain what changes when weights do not sum to one.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Each weighted contribution: [[1.  0. ]
#  [0.  1.2]
#  [0.4 0.4]] sum: [1.4 1.6]
# Changed mixture: [1.  2.6]
# ```
