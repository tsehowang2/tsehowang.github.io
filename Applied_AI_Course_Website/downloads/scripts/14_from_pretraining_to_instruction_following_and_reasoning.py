# # Tutorial 14: From pretraining to instruction following and reasoning
# 
# **Prerequisite:** lecture 14. **Compute:** CPU; no downloads for masking exercise.
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
# Explain why the first response token is predicted by the last prompt position after shifting.

# Instruction training is still token prediction, but the desired responses define supervision. We isolate label masking using a teaching whitespace tokenizer. Production model serialization and token boundaries must use that model’s real chat template.

# Explicit shift: logit at i predicts label at i+1.
prompt = 'User How do I go to school Assistant'
answer = 'Take the bus <eos>'
words = (prompt + ' ' + answer).split()
vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
ids = [vocab[w] for w in words]
n = len(prompt.split())
labels = [-100] * n + ids[n:]
shifted_inputs = ids[:-1]
shifted_targets = labels[1:]
assert shifted_targets[n - 1] == vocab['Take']
assert shifted_targets[-1] == vocab['<eos>'] and labels[-1] != -100
print(list(zip(words, ids, labels)))
print('First supervised pair:', words[n - 1], '->', words[n])
print('Last supervised pair: bus -> <eos>')


# ### Independent experiment
# Make six independently solvable arithmetic questions. Compare direct and concise-step prompts at equal budgets if a model is available. Use a calculator for correctness and preserve wrong rationales. Without a model, submit the test protocol and clearly label it unexecuted.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# [('User', 0, -100), ('How', 1, -100), ('do', 2, -100), ('I', 3, -100), ('go', 4, -100), ('to', 5, -100), ('school', 6, -100), ('Assistant', 7, -100), ('Take', 8, 8), ('the', 9, 9), ('bus', 10, 10), ('<eos>', 11, 11)]
# First supervised pair: Assistant -> Take
# Last supervised pair: bus -> <eos>
# ```
