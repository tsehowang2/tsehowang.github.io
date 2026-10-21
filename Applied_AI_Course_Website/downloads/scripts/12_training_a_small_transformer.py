# # Tutorial 12: Training a small transformer
# 
# **Prerequisite:** lecture 12. **Compute:** CPU or Colab; PyTorch.
# 
# Predict the result before each code cell. Run cells in order from a fresh kernel. Tutorials 01–16 run without coursekit. Tutorials 17–32 use the extracted full course pack linked on the tutorial page. Reference outputs are labeled below; keep your own actual results.

import random
import sys, json, math
from pathlib import Path
import sys, json, math, time, random
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
COURSE = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RUNS = COURSE / 'runs'
RUNS.mkdir(exist_ok=True)
random.seed(7)
print('Outputs directory:', RUNS)


# This is a tiny causal byte transformer trained on ordinary school and travel sentences. CPU steps check correctness; they cannot establish general language understanding. Read the model class one sublayer at a time before training.

import sys, json, math
import importlib.util, subprocess
if importlib.util.find_spec('torch') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==2.8.0'])
import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(7)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
T, D, H = (32, 64, 4)

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.n1, self.n2 = (nn.LayerNorm(D), nn.LayerNorm(D))
        self.attn = nn.MultiheadAttention(D, H, batch_first=True, dropout=0)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        h = self.n1(x)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), 1)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))

class TinyLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(256, D)
        self.pos = nn.Embedding(T, D)
        self.blocks = nn.Sequential(Block(), Block())
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 256)

    def forward(self, ids):
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        return self.head(self.norm(self.blocks(x)))
model = TinyLM().to(device)
print(device, 'parameters', sum((p.numel() for p in model.parameters())))


import json
import math
import sys, json, math
train_docs = [{'text': s} for s in ['I go to school by bus.', 'I go to school by train.', 'You go to work by bus.', 'We walk to school.']] * 20
dev_docs = [{'text': s} for s in ['She goes to school by bicycle.', 'They go to work by train.']] * 10

def encode(ds):
    return torch.tensor(list('\n'.join((d['text'] for d in ds)).encode('utf-8')), dtype=torch.long)
train, dev = (encode(train_docs), encode(dev_docs))

def batch(data, B=4):
    starts = torch.randint(len(data) - T, (B,))
    xs = torch.stack([data[i:i + T] for i in starts])
    ys = torch.stack([data[i + 1:i + T + 1] for i in starts])
    return (xs.to(device), ys.to(device))
x, y = batch(train)
assert torch.equal(x[0, 1:], y[0, :-1])
model.eval()
with torch.no_grad():
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % 256
    assert torch.allclose(model(x)[:, :-1], model(x2)[:, :-1], atol=1e-05)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
history = []
steps = 20
for step in range(steps):
    x, y = batch(train)
    loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    history.append({'step': step, 'train_loss': loss.item()})
    print(history[-1])
model.eval()
with torch.no_grad():
    x, y = batch(dev)
    dev_loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item()
print('Single sampled dev loss (not stable estimate)', dev_loss, 'perplexity', math.exp(dev_loss))
torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else [], 'step': steps}, RUNS / 'tiny_lm.pt')
(RUNS / 'week12_config.json').write_text(json.dumps({'T': T, 'D': D, 'H': H, 'blocks': 2, 'seed': 7, 'torch': torch.__version__, 'device': device, 'history': history, 'dev_loss': dev_loss}, indent=2), encoding='utf-8')


# Evaluate multiple dev batches, run a one-batch overfit diagnostic and try one architecture ablation at equal token/step budget. Implement and verify checkpoint resume as independent work.

# ### Reference output
# Expected runtime checks: 20 training-history entries on either CPU or GPU, shifted targets, causal invariance, finite loss, and saved checkpoint/config. Numeric losses vary.
