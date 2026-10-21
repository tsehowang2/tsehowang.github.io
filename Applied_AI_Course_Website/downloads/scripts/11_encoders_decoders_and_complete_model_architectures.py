# # Tutorial 11: Encoders decoders and complete model architectures
# 
# **Prerequisite:** lecture 11. **Compute:** CPU; standard library and NumPy.
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
# For the third decoder position, list which source and target positions are visible.

# Encoder self attention sees both directions; a decoder generating text sees only the prefix. Cross attention uses decoder queries with encoder keys and values, so the query and key lengths can differ.

import numpy as np
T, S, D = (6, 4, 8)
encoder_mask = np.ones((S, S), dtype=bool)
decoder_mask = np.tril(np.ones((T, T), dtype=bool))
cross_mask = np.ones((T, S), dtype=bool)
print('Encoder visibility:', encoder_mask.astype(int), 'Decoder visibility:', decoder_mask.astype(int), 'Cross visibility:', cross_mask.astype(int), sep='\n')


# ### Test visibility with changing values
# Perturb one source row, then one future target row. Random values make a wrong mask observable.

import numpy as np
rng = np.random.default_rng(7)
target = rng.normal(size=(6, 8))
source = rng.normal(size=(4, 8))

def attend(q, k, v, mask):
    scores = np.where(mask, q @ k.T / np.sqrt(q.shape[-1]), -np.inf)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True) @ v
causal = np.tril(np.ones((6, 6), dtype=bool))
changed = target.copy()
changed[-1] += 7
original = attend(target, target, target, causal)
modified = attend(changed, changed, changed, causal)
assert np.allclose(original[:-1], modified[:-1])
assert not np.allclose(original[-1], modified[-1])
unmasked = np.ones((6, 6), dtype=bool)
assert not np.allclose(attend(target, target, target, unmasked)[:-1], attend(changed, changed, changed, unmasked)[:-1])
visible = np.ones((6, 4), dtype=bool)
changed_source = source.copy()
changed_source[0] += 7
cross = attend(target, source, source, visible)
cross_changed = attend(target, changed_source, changed_source, visible)
assert np.all(np.linalg.norm(cross - cross_changed, axis=1) > 1e-08)
assert np.allclose(cross[:1], attend(target[:1], source, source, visible[:1]))
print('Causal invariance, unmasked control, source sensitivity, query-row independence: passed')


# ### Independent experiment
# Write a shape table for a batch of two with three heads. Explain why changing an encoder source token may change every target output while changing a future decoder token must not change earlier outputs.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Encoder visibility:
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]
# Decoder visibility:
# [[1 0 0 0 0 0]
#  [1 1 0 0 0 0]
#  [1 1 1 0 0 0]
#  [1 1 1 1 0 0]
#  [1 1 1 1 1 0]
#  [1 1 1 1 1 1]]
# Cross visibility:
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]
# Causal invariance, unmasked control, source sensitivity, query-row independence: passed
# ```
