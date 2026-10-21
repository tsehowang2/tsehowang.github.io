# # Tutorial 21: Evaluating retrieval answers and uncertainty
# 
# **Prerequisite:** lecture 21. **Compute:** CPU; no downloads.
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


# A test harness repeats the same cases and saves results. Read one query, its evidence label and an actual extractor output before computing aggregate metrics. Synthetic uncertainty examples teach the method but are not your measured improvement.

import json
tasks = json.loads((COURSE / 'data/queries.json').read_text(encoding='utf-8'))
index = LexicalIndex(load_docs())
records = []
for t in tasks:
    result = extract_answer(t['question'], index, t['version'])
    records.append({'task_id': t['id'], 'group': t['group'], 'expected': t['relevant'], 'result': result, 'retrieval': retrieval_metrics(result['citations'], t['relevant'], 3), 'manual_correct': None, 'manual_supported': None, 'manual_refusal_appropriate': None})
(RUNS / 'week21_to_grade.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
assert retrieval_metrics(['C', 'A', 'D'], ['A', 'B'], 3) == {'recall': 0.5, 'hit': 1, 'rr': 0.5}
print('Illustrative paired data, NOT model results:', paired_bootstrap([1, 1, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0]))
print('Fill manual fields; preserve failures and evaluate unsupported cases separately.')


# Add five original adversarial tasks and a predeclared primary metric. Compare actual configurations and replace illustrative arrays with measured results. Inspect group dependence before resampling individual tasks. Null grades mean ungraded, not success or zero.

# ### Graded ranking and paired outcomes
# nDCG needs a declared gain convention and ideal ranking. The McNemar example uses ten discordant paired tasks: A alone succeeds once and B alone succeeds nine times. Its exact binomial calculation assumes independent paired units; duplicated task families violate that assumption.

import math
labels = [3, 1, 2]

def dcg(values):
    return sum(((2 ** rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(values)))
ndcg = dcg(labels) / dcg(sorted(labels, reverse=True))
assert 0 < ndcg <= 1
b, c = (1, 9)
n = b + c
p_exact = min(1.0, 2 * sum((math.comb(n, i) for i in range(min(b, c) + 1))) / 2 ** n)
assert math.isclose(p_exact, 0.021484375)
print('nDCG using gain 2**rel-1:', ndcg, 'Exact two-sided McNemar p:', p_exact)


# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Illustrative paired data, NOT model results: {'difference': 0.16666666666666666, 'ci95': [0.0, 0.5], 'n': 6, 'seed': 7}
# Fill manual fields; preserve failures and evaluate unsupported cases separately.
# nDCG using gain 2**rel-1: 0.9721212198129313 Exact two-sided McNemar p: 0.021484375
# ```
