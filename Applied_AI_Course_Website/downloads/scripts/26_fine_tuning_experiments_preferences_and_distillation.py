# # Tutorial 26: Fine tuning experiments preferences and distillation
# 
# **Prerequisite:** lecture 26. **Compute:** Colab GPU preferred; CPU two-step pilot is slow.
# 
# Predict the result before each code cell. Run cells in order from a fresh kernel. Tutorials 01–16 run without coursekit. Tutorials 17–32 use the extracted full course pack linked on the tutorial page. Reference outputs are labeled below; keep your own actual results.

import random
from pathlib import Path
import sys, json, time, random, hashlib, math
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
candidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / 'llm_course', Path.cwd() / 'llm_course_v2', Path('/content/llm_course'), Path('/content/llm_course_v2')]
COURSE = next((p for p in candidates if (p / 'coursekit/core.py').exists()), None)
if COURSE is None:
    raise RuntimeError('Extract the complete course pack; run from its root or notebooks directory.')
if str(COURSE) not in sys.path:
    sys.path.insert(0, str(COURSE))
RUNS = COURSE / 'runs'
RUNS.mkdir(exist_ok=True)
from coursekit.core import *
random.seed(7)
print('Course root:', COURSE)


# ### Current generator recipe
# The small instruction model is Qwen/Qwen3-0.6B. This recipe pins Transformers 4.56.2 and PyTorch 2.8.0, and sets enable_thinking=False in the chat template. Resolve and pin the same public model revision for tokenizer and weights. Model downloads and GPU training are not validated by the CPU checks.

# ### DPO scale calculation before training
# Use beta=0.1 and relative margin=0.3. The scaled margin is 0.03, giving loss about 0.67826. This is a common implementation-default scale, not a universal best setting. Run this arithmetic cell without downloading a model.

import math
beta = 0.1
relative_margin = 0.4 - 0.1
scaled_margin = beta * relative_margin
dpo_loss = math.log1p(math.exp(-scaled_margin))
assert abs(dpo_loss - 0.6782596763414483) < 1e-12
print('Illustrative DPO loss (not DPO training)', dpo_loss)


# ### Distillation on a visible teacher distribution
# A teacher supplies soft probabilities. Calculate cross-entropy and KL, then take one gradient step on student logits. This is a three-class teaching example.

import numpy as np
teacher = np.array([0.7, 0.2, 0.1])
student_logits = np.zeros(3)

def sm(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def distill_loss(z):
    return -np.sum(teacher * np.log(sm(z)))
before = distill_loss(student_logits)
student_logits -= 0.5 * (sm(student_logits) - teacher)
after = distill_loss(student_logits)
kl = after + np.sum(teacher * np.log(teacher))
assert after < before and kl >= 0
print('Teacher:', teacher, 'Student after step:', sm(student_logits).round(4))
print('Distillation CE before/after:', before, after, 'KL:', kl)
print('Training LoRA alpha/r:', 8 / 4)


# Begin with the week-25 factorization and the week-24 audited targets. The following pilot trains actual LoRA weights. Understand masking, frozen weights and trainable parameters before starting the optimizer. Preserve negative results.

# Float32 keeps this beginner training path simple. Use a small model and short input.
import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.8.0', 'transformers==4.56.2', 'peft>=0.13,<1'])
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
if not (RUNS / 'week24_train.json').exists():
    raise RuntimeError('Run tutorial 24 first.')
torch.manual_seed(7)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = 'Qwen/Qwen3-0.6B'
from huggingface_hub import model_info
resolved_revision = model_info(MODEL_ID).sha
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=resolved_revision)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=resolved_revision, dtype=torch.float32).to(device)
base.config.use_cache = False
base_revision = resolved_revision
model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
model.print_trainable_parameters()


import json
import time

def prepare(record, max_length=2048):
    messages = record['messages']
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    # Qwen3-specific completion: generation prefix includes its empty think block.
    # Supervise the response and the tokenizer's response-ending EOS, not the prompt.
    response = tokenizer.encode(messages[-1]['content'], add_special_tokens=False)
    response.append(tokenizer.eos_token_id)
    full = prefix + response
    if len(full) > max_length:
        raise ValueError('Example exceeds budget: shorten evidence without dropping response/EOS.')
    ids = full
    labels = [-100] * len(prefix) + response
    assert labels[-1] == tokenizer.eos_token_id
    assert sum(value != -100 for value in labels[1:]) > 0
    return {'input_ids': torch.tensor([ids], device=device), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long, device=device), 'labels': torch.tensor([labels], device=device)}
train = json.loads((RUNS / 'week24_train.json').read_text(encoding='utf-8'))
dev = json.loads((RUNS / 'week24_dev.json').read_text(encoding='utf-8'))
train_batches = [prepare(r) for r in train]
dev_batches = [prepare(r) for r in dev]

def score(batches, disable=False):
    model.eval()
    total, n = (0.0, 0)
    context = model.disable_adapter() if disable else __import__('contextlib').nullcontext()
    with context, torch.no_grad():
        for b in batches:
            count = int((b['labels'][:, 1:] != -100).sum())
            total += model(**b).loss.item() * count
            n += count
    return total / n
before = score(dev_batches, disable=True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0001)
steps = 20 if device == 'cuda' else 2
if device == 'cuda':
    torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
losses = []
model.train()
for step in range(steps):
    b = train_batches[step % len(train_batches)]
    optimizer.zero_grad()
    loss = model(**b).loss
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    loss.backward()
    if step == 0:
        assert any((p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad))
        assert all((p.grad is None for p in model.parameters() if not p.requires_grad))
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    losses.append(loss.item())
    print(step, loss.item())
after = score(dev_batches)
model.eval()
with torch.no_grad():
    expected_logits = model(**{k: v for k, v in dev_batches[0].items() if k != 'labels'}).logits.detach().cpu()
adapter = RUNS / 'week26_adapter'
model.save_pretrained(adapter)
tokenizer.save_pretrained(adapter)
report = {'base_model': MODEL_ID, 'base_revision': base_revision, 'steps': steps, 'device': device, 'base_dev_nll': before, 'adapter_dev_nll': after, 'losses': losses, 'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None}
(RUNS / 'week26_training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(report)


# Release the first model before reloading to avoid two full GPU copies.
del optimizer, train_batches, dev_batches, model, base, loss
import gc
gc.collect()
if device == 'cuda':
    torch.cuda.empty_cache()
reloaded_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=base_revision, dtype=torch.float32).to(device)
reloaded = PeftModel.from_pretrained(reloaded_base, adapter).eval()
sample = prepare(dev[0])
with torch.no_grad():
    reload_output = reloaded(**sample)
assert torch.allclose(expected_logits, reload_output.logits.cpu(), rtol=1e-05, atol=1e-05)
print('Reload logits match saved adapter; loss:', reload_output.loss.item())


# Finish by generating base/adapter outputs on independent task cases with fixed decoding and the same parser. NLL above is a training diagnostic. Two CPU steps demonstrate mechanics, not useful adaptation. Save resolved packages/config/data hashes. QLoRA requires an explicitly checked backend and is optional.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Illustrative DPO loss: 0.6782596763414483
# Student after distillation step: [0.3969 0.3091 0.2940]
# Distillation CE: 1.0986122886681098 -> 1.004107701106269
# KL: 0.20228914856293168
# LoRA alpha/r: 2.0
# Model training and reload: not run in this revision.
# ```
