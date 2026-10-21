# # Tutorial 03: What word vectors can represent
# 
# **Prerequisite:** lecture 03. **Compute:** CPU; standard library and NumPy.
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
# Predict what happens to dot product and cosine when train is multiplied by five.

# These two-dimensional vectors are invented to make the arithmetic visible. They are not trained embeddings. Scale one vector while keeping its direction and compare two similarity measures.

import numpy as np
vectors = {'bus': np.array([1.0, 0.0]), 'train': np.array([0.8, 0.6]), 'banana': np.array([0.0, 1.0])}
q = vectors['bus']
for word, v in vectors.items():
    cosine = q @ v / (np.linalg.norm(q) * np.linalg.norm(v))
    print(word, 'dot', q @ v, 'cosine', cosine)
v = 5 * vectors['train']
assert np.isclose(q @ v, 4)
assert np.isclose(q @ v / (np.linalg.norm(q) * np.linalg.norm(v)), 0.8)


# ### When cosine and dot product disagree
# Use the same q, a, and b as the lecture. b is longer but less aligned with q. Predict which candidate each scoring rule prefers before running.

import math
q = [1.0, 0.0]
a = [0.8, 0.6]
b = [3.0, 4.0]

def dot(x, y):
    return sum((u * v for u, v in zip(x, y)))

def cosine(x, y):
    return dot(x, y) / math.sqrt(dot(x, x) * dot(y, y))
assert dot(q, b) > dot(q, a) and cosine(q, a) > cosine(q, b)
print('Dot a,b:', dot(q, a), dot(q, b), 'Cosine a,b:', cosine(q, a), cosine(q, b))


# ### Independent experiment
# Create a random vocabulary table and look up bus by row ID. Permute the vocabulary IDs and table rows together; demonstrate that the retrieved vector stays the same. Explain why independently trained embedding spaces cannot be mixed.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# bus dot 1.0 cosine 1.0
# train dot 0.8 cosine 0.8
# banana dot 0.0 cosine 0.0
# Dot a,b: 0.8 3.0 Cosine a,b: 0.8 0.6
# ```
