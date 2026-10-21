# # Tutorial 01: What a language model is trying to learn
# 
# **Prerequisite:** lecture 01. **Compute:** CPU; no downloads.
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
# Add two train sentences by hand. Predict the new denominator before running.

# We first build a language model without a neural network. Each sentence is visible. Count only continuations of the same prefix, then turn counts into probabilities. These proportions describe this toy collection, not real commuting behavior.

from collections import Counter
sentences = ['I go to school by bus'] * 3 + ['I go to school by train', 'I go to school by bicycle']
vocabulary = ['bus', 'train', 'bicycle']

def continuation_probs(prefix, alpha=0.0):
    observed = Counter((s[len(prefix):] for s in sentences if s.startswith(prefix)))
    total = sum(observed.values())
    if total == 0 and alpha == 0:
        return None
    return {w: (observed[w] + alpha) / (total + alpha * len(vocabulary)) for w in vocabulary}
prefix = 'I go to school by '
counts = Counter((s[len(prefix):] for s in sentences if s.startswith(prefix)))
probs = continuation_probs(prefix)
print(counts, probs)
assert probs == {'bus': 0.6, 'train': 0.2, 'bicycle': 0.2}
assert continuation_probs('I go to work by ') is None
fallback = continuation_probs('I go to work by ', 1)
assert fallback == dict.fromkeys(vocabulary, 1 / 3)
print('Unseen prefix:', None, 'Add-one fallback:', fallback)


counts['train'] += 2
print({w: c / sum(counts.values()) for w, c in counts.items()})
assert sum(counts.values()) == 7
print('Three-token chain probability:', 0.5 * 0.4 * 0.25)


# ### Independent experiment
# Change the prefix to I go to work by. Explain why no observations is different from evidence that every completion is impossible.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Counter({'bus': 3, 'train': 1, 'bicycle': 1}) {'bus': 0.6, 'train': 0.2, 'bicycle': 0.2}
# Unseen prefix: None Add-one fallback: {'bus': 0.3333333333333333, 'train': 0.3333333333333333, 'bicycle': 0.3333333333333333}
# {'bus': 0.42857142857142855, 'train': 0.42857142857142855, 'bicycle': 0.14285714285714285}
# Three-token chain probability: 0.05
# ```
