# # Tutorial 13: Generation decoding and the KV cache
# 
# **Prerequisite:** lecture 13. **Compute:** CPU; standard library and NumPy.
# 
# Predict the result before each code cell. Run cells in order from a fresh kernel. Tutorials 01–16 run without coursekit. Tutorials 17–32 use the extracted full course pack linked on the tutorial page. Reference outputs are labeled below; keep your own actual results.

import random
import sys
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
# Explain why the most likely first token need not lead to the most likely complete sequence.

# ### Current generator recipe
# The small instruction model is Qwen/Qwen3-0.6B. This recipe pins Transformers 4.56.2 and PyTorch 2.8.0, and sets enable_thinking=False in the chat template. Resolve and pin the same public model revision for tokenizer and weights. Model downloads and GPU training are not validated by the CPU checks.

# First perform greedy versus seeded sampling on a three-word vocabulary. This isolates decoding from training. The optional next cell uses a real pretrained model to complete an ordinary school sentence; it introduces no retrieval system.

import sys
import numpy as np
p = np.array([0.6, 0.2, 0.2])
words = np.array(['bus', 'train', 'bicycle'])
rng = np.random.default_rng(7)
print('Greedy:', words[p.argmax()])
print('Twenty samples:', rng.choice(words, 20, p=p))
for temperature in [0.5, 1.0, 2.0]:
    z = np.log(p) / temperature
    e = np.exp(z - z.max())
    print(temperature, e / e.sum())


import sys
RUN_MODEL = False
if RUN_MODEL:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==4.56.2', 'torch==2.8.0'])
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = 'Qwen/Qwen3-0.6B'
    from huggingface_hub import model_info
    resolved_revision = model_info(model_id).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved_revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=resolved_revision).eval()
    messages = [{'role': 'user', 'content': 'Complete with one transport word: I go to school by'}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors='pt')
    tokens = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    with torch.no_grad():
        out = model.generate(**tokens, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print('Revision', resolved_revision, 'IDs', tokens['input_ids'].tolist())
    print(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
else:
    print('Real-model generation skipped. Set RUN_MODEL=True to download and run.')


# ### Decoding filters and a cached computation
# Compare two-step sequence probabilities, then cached and uncached single-layer causal attention. This NumPy check stores K/V explicitly; it does not measure a pretrained model.

import numpy as np
p = np.array([0.5, 0.3, 0.15, 0.05])

def retain(p, indices):
    out = np.zeros_like(p)
    out[indices] = p[indices]
    return out / out.sum()
top_k = retain(p, np.argsort(-p)[:2])
order = np.argsort(-p)
stop = np.searchsorted(np.cumsum(p[order]), 0.7) + 1
top_p = retain(p, order[:stop])
assert np.allclose(top_k, [0.625, 0.375, 0, 0]) and np.allclose(top_p, top_k)
greedy_sequence = 0.6 * 0.5
alternative_sequence = 0.4 * 0.9
assert alternative_sequence > greedy_sequence
print('top-k=2:', top_k, 'top-p=.7:', top_p)
print('Greedy A->end:', greedy_sequence, 'B->end:', alternative_sequence)
rng = np.random.default_rng(7)
x = rng.normal(size=(5, 4))
wq, wk, wv = [rng.normal(size=(4, 3)) for _ in range(3)]
cache_k, cache_v = ([], [])
for t in range(len(x)):
    q = x[t:t + 1] @ wq
    cache_k.append(x[t:t + 1] @ wk)
    cache_v.append(x[t:t + 1] @ wv)

    def read(k, v):
        scores = q @ k.T / np.sqrt(3)
        e = np.exp(scores - scores.max())
        return e / e.sum() @ v
    cached = read(np.concatenate(cache_k), np.concatenate(cache_v))
    uncached = read(x[:t + 1] @ wk, x[:t + 1] @ wv)
    assert np.allclose(cached, uncached, atol=1e-12)
print('Cache equality: 5/5 steps; projected K/V rows: cached=5, uncached=15')


# ### Independent experiment
# For a real-model run, compare greedy and two seeded samples under the same prefix and maximum length. Record model revision, actual tokens and latency. Explain how a KV cache changes computation but not the intended conditional distribution.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Greedy: bus
# Twenty samples: ['train' 'bicycle' 'train' 'bus' 'bus' 'bicycle' 'bus' 'bicycle' 'train'
#  'bus' 'bus' 'bus' 'bus' 'bus' 'bus' 'bus' 'bicycle' 'train' 'train'
#  'bicycle']
# 0.5 [0.81818182 0.09090909 0.09090909]
# 1.0 [0.6 0.2 0.2]
# 2.0 [0.46410162 0.26794919 0.26794919]
# Real-model generation skipped. Set RUN_MODEL=True to download and run.
# top-k=2: [0.625 0.375 0.    0.   ] top-p=.7: [0.625 0.375 0.    0.   ]
# Greedy A->end: 0.3 B->end: 0.36000000000000004
# Cache equality: 5/5 steps; projected K/V rows: cached=5, uncached=15
# ```
