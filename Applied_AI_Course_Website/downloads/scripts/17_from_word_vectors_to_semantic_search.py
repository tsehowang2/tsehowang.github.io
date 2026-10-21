# # Tutorial 17: From word vectors to semantic search
# 
# **Prerequisite:** lecture 17. **Compute:** CPU; standard library and NumPy.
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


# ### Pause before the next step
# Label the relevant document yourself. Explain why this hand-designed geometry cannot prove that an embedding model understands the query.

# ### Current embedding recipe
# Use intfloat/multilingual-e5-small with query: before every query and passage: before every passage, including Chinese text. The output has 384 dimensions and the model input limit is 512 tokens; prefixes and special tokens use part of that budget. Rebuild the collection when the embedding recipe changes.

# Before a vector database, semantic search is a list of document vectors and a query vector. Begin with school-handbook sentences whose relevance you can judge yourself. A sentence embedding model is trained for sequence representation; it is not simply any hidden-state table.

# Invented normalized geometry to inspect exact search, NOT model embeddings.
texts = ['The school bus leaves at eight.', 'Lunch starts at noon.', 'The library closes at five.']
query = 'At what time does the morning transport depart?'
print('Query:', query)
for i, t in enumerate(texts):
    print(i, t)
import numpy as np
vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
q = np.array([1.0, 0.0])
scores = vectors @ q
print('Illustrative ranking:', np.argsort(-scores).tolist())


# ### Calculate the contrastive example
# Use the invented score matrix from lecture 17. The diagonal contains the labeled positives. Predict the mean loss before running the cell. Then compare the Q1 loss when another useful passage is incorrectly kept as a negative. This cell calculates an objective; it does not train an embedding model.

import math
scores = [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0]]

def candidate_probs(row):
    exps = [math.exp(x - max(row)) for x in row]
    return [x / sum(exps) for x in exps]
probs = [candidate_probs(row) for row in scores]
losses = [-math.log(row[i]) for i, row in enumerate(probs)]
print('Candidate probabilities:', probs)
print('Mean contrastive loss:', sum(losses) / len(losses))
print('False-negative example Q1 loss:', -math.log(candidate_probs([2.0, 2.0, 0.0])[0]))


import sys
RUN_EMBEDDINGS = False
if RUN_EMBEDDINGS:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sentence-transformers>=3,<6'])
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('intfloat/multilingual-e5-small')
    matrix = encoder.encode(['passage: ' + t for t in texts], normalize_embeddings=True)
    q = encoder.encode(['query: ' + query], normalize_embeddings=True)[0]
    print('Actual model scores:', matrix @ q)
    print('Actual ranking:', np.argsort(-(matrix @ q)).tolist())
else:
    print('Actual embedding model skipped; enable RUN_EMBEDDINGS for model results.')


# ### Cosine scores and temperature
# The earlier [2,1,0] values are teaching logits. Here the inputs are possible cosine scores, divided by a chosen temperature before softmax. Lower loss after rescaling this fixed example is not evidence that an embedding model improved.

import math
cosines = [0.8, 0.6, 0.2]

def probs_at_temperature(xs, tau):
    logits = [x / tau for x in xs]
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    return [x / sum(e) for x in e]
for tau in [1.0, 0.05]:
    p = probs_at_temperature(cosines, tau)
    print('tau', tau, 'p', p, 'positive loss', -math.log(p[0]))
assert -math.log(probs_at_temperature(cosines, 0.05)[0]) < -math.log(probs_at_temperature(cosines, 1.0)[0])


# ### Independent experiment
# Run the optional real embedding cell on six original paraphrases, compare lexical overlap and dense ranking, and report every miss. Record the model revision; do not call the invented vectors model results.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Query: At what time does the morning transport depart?
# 0 The school bus leaves at eight.
# 1 Lunch starts at noon.
# 2 The library closes at five.
# Illustrative ranking: [0, 2, 1]
# Candidate probabilities: [[0.6652409557748218, 0.24472847105479764, 0.09003057317038046], [0.09003057317038046, 0.6652409557748218, 0.24472847105479764], [0.24472847105479764, 0.09003057317038046, 0.6652409557748218]]
# Mean contrastive loss: 0.4076059644443804
# False-negative example Q1 loss: 0.7586236756795135
# Actual embedding model skipped; enable RUN_EMBEDDINGS for model results.
# tau 1.0 p [0.42237892110127173, 0.34581461215750964, 0.23180646674121863] positive loss 0.8618524505311191
# tau 0.05 p [0.9820078648958167, 0.01798610143932861, 6.033664854558337e-06] positive loss 0.01815596160086702
# ```
