# # Tutorial 16: A complete transformer walkthrough and first semester review
# 
# **Prerequisite:** lecture 16. **Compute:** CPU worksheet; PyTorch only for the week-12 reproduction.
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
# Explain how a six-position input can have six next-token targets when taken from a longer sequence, although a standalone six-token sentence only supplies five adjacent pairs.

# This is a consolidation worksheet. We put dimensions and operation counts beside a complete model path. The arithmetic does not replace the actual tiny-model run from week 12. Bring its report and explain each stage.

B, T, D, H, V = (2, 6, 8, 2, 20)
dh = D // H
trace = {'token IDs': (B, T), 'embeddings': (B, T, D), 'split Q/K/V': (B, H, T, dh), 'scores': (B, H, T, T), 'merged attention': (B, T, D), 'vocabulary logits': (B, T, V), 'shifted targets': (B, T)}
for name, shape in trace.items():
    print(name, shape)
score_bytes = B * H * T * T * 4
assert score_bytes == 576
print('Explicit float32 scores in this example:', score_bytes, 'bytes')
print('Use the week-12 implementation to verify these shapes on a live model.')


# ### Count the matrix multiplications
# Count multiply-accumulates (MACs); each is approximately two FLOPs. Ignore normalization, softmax, biases and activations.

B, T, d_model, H, V = (2, 6, 8, 2, 20)
d_head = d_model // H
counts = {'QKV': 3 * B * T * d_model * d_model, 'QK and weights-V': 2 * B * H * T * T * d_head, 'output projection': B * T * d_model * d_model, 'MLP': 8 * B * T * d_model * d_model, 'vocabulary': B * T * d_model * V}
for name, macs in counts.items():
    print(name, macs, 'MACs')
print('Total:', sum(counts.values()), 'MACs; approximate FLOPs:', 2 * sum(counts.values()))
assert sum(counts.values()) == 12288


# ### Independent experiment
# Reproduce the week-12 model in a fresh environment and defend one controlled ablation. Complete semester exam 1 instead of the usual independent experiment. Record corrections rather than adding extra hours.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# token IDs (2, 6)
# embeddings (2, 6, 8)
# split Q/K/V (2, 2, 6, 4)
# scores (2, 2, 6, 6)
# merged attention (2, 6, 8)
# vocabulary logits (2, 6, 20)
# shifted targets (2, 6)
# Explicit float32 scores in this example: 576 bytes
# Use the week-12 implementation to verify these shapes on a live model.
# QKV 2304 MACs
# QK and weights-V 1152 MACs
# output projection 768 MACs
# MLP 6144 MACs
# vocabulary 1920 MACs
# Total: 12288 MACs; approximate FLOPs: 24576
# ```
