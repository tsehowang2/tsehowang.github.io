# # Tutorial 19: RAG as an open book question
# 
# **Prerequisite:** lecture 19. **Compute:** CPU; no downloads.
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
# Write a supported answer manually and attach the source ID. Ask an absent question and explain why the generator must not fill in missing facts.

# RAG is an open-book question. We first inspect evidence and build a prompt; no generation is claimed. Then a deterministic extractor shows a simple documentation baseline. Week 20 contains the optional full real-model RAG path.

handbook = [{'id': 'bus', 'title': 'Transport', 'version': '2', 'source': 'handbook', 'text': 'The school bus leaves at 8:00.'}, {'id': 'lunch', 'title': 'Meals', 'version': '2', 'source': 'handbook', 'text': 'Lunch starts at 12:00.'}]
handbook_index = LexicalIndex(handbook)
question = 'When does the school bus leave?'
evidence = handbook_index.search(question, '2', 1)
prompt = 'Use only the evidence. Cite IDs. If missing, say insufficient evidence.\n'
prompt += '\n'.join(('[' + d['id'] + '] ' + d['text'] for d in evidence))
prompt += '\nQuestion: ' + question
print(prompt)
assert evidence and evidence[0]['id'] == 'bus'
miss = handbook_index.search('Pupils boarding hour?', '2', 1)
assert miss == []
print('Paraphrase retrieval miss:', miss)
index = LexicalIndex(load_docs())
print('Deterministic extractor, not an LLM:', extract_answer('default request timeout', index, '2'))


# ### Independent experiment
# Add a contradictory older handbook and a version field. Show the exact evidence chosen for each version. Compare failure at retrieval, packing and generation; do not treat a correctly formatted citation as proof of support.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Use only the evidence. Cite IDs. If missing, say insufficient evidence.
# [bus] The school bus leaves at 8:00.
# Question: When does the school bus leave?
# Paraphrase retrieval miss: []
# Deterministic extractor, not an LLM: {'answer': 'Atlas SDK version 2 uses timeout_ms. The default request timeout is 30000 milliseconds. Set timeout_ms when creating the Client.\nThe example client allows at most 4 simultaneous requests by default. Additional work should wait in a bounded queue. This statement concerns request concurrency, not the retry count.\nAtlas SDK version 2 retries transient errors at most 3 times by default. For this example, the first retry waits 100 milliseconds, the next 200 milliseconds, then 400 milliseconds. There is no jitter in this teaching fixture.', 'citations': ['timeout-v2', 'concurrency-v2', 'retry-v2'], 'version': '2', 'status': 'evidence_found', 'generator': 'lexical_extractor_not_llm', 'grounding_verified': False, 'latency_ms': 0.03910018131136894}
# ```
