# # Tutorial 02: From text to tokens and numerical IDs
# 
# **Prerequisite:** lecture 02. **Compute:** CPU; no downloads.
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
# Write which word is the target at each position. Why does the final bus not appear in the input for this single training pair?

# For this first tokenizer, words are separated by spaces. A vocabulary maps each word to an arbitrary integer. The integer identifies a row; it is not a numerical meaning. We will explicitly compare this teaching tokenizer with subword tokenization later.

sentence = 'I go to school by bus'
words = sentence.split()
vocab = {w: i for i, w in enumerate(words)}
ids = [vocab[w] for w in words]
print(vocab, ids)
inputs, targets = (ids[:-1], ids[1:])
print('Inputs:', inputs, 'Targets:', targets)
assert inputs[1:] == targets[:-1]
print('Characters:', len(sentence), 'UTF-8 bytes:', len(sentence.encode()))


# ### A BPE merge with unequal counts
# Count adjacent pairs over bus four times, busy twice, and plus once. The underscore is a visible word-end marker in this teaching tokenizer. It is not a rule imposed by every production tokenizer.

from collections import Counter
corpus = [('b', 'u', 's', '_')] * 4 + [('b', 'u', 's', 'y', '_')] * 2 + [('p', 'l', 'u', 's', '_')]
counts = Counter((pair for word in corpus for pair in zip(word, word[1:])))
print('Pair counts:', counts)
assert counts['u', 's'] == 7 and counts['b', 'u'] == 6
winner = max(counts, key=counts.get)
assert winner == ('u', 's')

def merge(word, pair):
    result = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            result.append(word[i] + word[i + 1])
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)
merged = [merge(w, winner) for w in corpus]
next_counts = Counter((pair for word in merged for pair in zip(word, word[1:])))
assert next_counts['b', 'us'] == 6
print('First merge:', winner, 'Next best:', max(next_counts, key=next_counts.get))


# ### Measure multilingual token lengths
# Optionally tokenize the same travel meaning in English and Traditional Chinese, plus whitespace and number variants. Record the tokenizer revision and actual counts. These measurements depend on the chosen tokenizer; UTF-8 bytes and model tokens are different units.

import sys
RUN_TOKENIZER = False
if RUN_TOKENIZER:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==4.56.2'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    examples = ['I go to school by bus.', '我乘巴士上學。', 'bus', ' bus', 'bus  bus', '257', '1,257', '１２３']
    for text in examples:
        ids = tokenizer.encode(text, add_special_tokens=False)
        print(repr(text), 'UTF-8 bytes', len(text.encode('utf-8')), 'model tokens', len(ids), 'pieces', tokenizer.convert_ids_to_tokens(ids))
    print('Tokenizer revision', tokenizer.init_kwargs.get('_commit_hash'))
else:
    print('Tokenizer measurement skipped; no token counts are assumed.')


# ### Independent experiment
# Compare school and school. in the whitespace vocabulary. Add punctuation handling, then explain how changing a tokenizer requires keeping the embedding table consistent.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# {'I': 0, 'go': 1, 'to': 2, 'school': 3, 'by': 4, 'bus': 5} [0, 1, 2, 3, 4, 5]
# Inputs: [0, 1, 2, 3, 4] Targets: [1, 2, 3, 4, 5]
# Characters: 21 UTF-8 bytes: 21
# Pair counts: Counter({('u', 's'): 7, ('b', 'u'): 6, ('s', '_'): 5, ('s', 'y'): 2, ('y', '_'): 2, ('p', 'l'): 1, ('l', 'u'): 1})
# First merge: ('u', 's') Next best: ('b', 'us')
# Tokenizer measurement skipped; no token counts are assumed.
# ```
