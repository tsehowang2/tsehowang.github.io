# # Tutorial 24: Fine tuning starts with a teaching dataset
# 
# **Prerequisite:** lecture 24. **Compute:** CPU; no downloads.
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


# A supervised example is a prompt and a desired response. Before preparing technical targets, write one ordinary instruction such as asking how to travel to school. Mark which tokens carry response loss. Then audit the supplied documentation records.

# Include eligible distractors; never select context using the gold labels.
import json
import random
docs = load_docs()
byid = {d['id']: d for d in docs}
records = []
answers = {'q01': 'The default request timeout is 30000 milliseconds.', 'q02': 'The default request timeout is 10000 milliseconds.', 'q03': 'The first retry backoff is 100 milliseconds.', 'q04': 'The API token is held in ATLAS_TOKEN.', 'q05': 'The allowed timeout_ms range is 1 to 120000.', 'q06': 'Invalidate cached responses after source deletion.', 'q07': 'Use next_cursor to fetch the next pagination results.', 'q08': 'Insufficient evidence.', 'q09': 'The default simultaneous request limit is 4.', 'q10': 'Version 1 uses 10000 milliseconds; version 2 uses 30000 milliseconds.', 'q11': 'Insufficient evidence.', 'q12': 'AUTH_EXPIRED indicates an expired token.'}
for t in json.loads((COURSE / 'data/queries.json').read_text(encoding='utf-8')):
    candidates = [d for d in docs if t['version'] == 'all' or d['version'] == t['version']]
    context = '\n'.join(('[' + d['id'] + '] version=' + d['version'] + ' ' + d['text'] for d in candidates))
    target = {'answer': answers[t['id']], 'citations': t['relevant']}
    records.append({'task_id': t['id'], 'group': t['group'], 'version': t['version'], 'messages': [{'role': 'user', 'content': 'Use this evidence only:\n' + context + '\nQuestion: ' + t['question']}, {'role': 'assistant', 'content': json.dumps(target)}]})
assert len(records) == 12 and any((r['version'] == 'all' for r in records))
groups = sorted({r['group'] for r in records})
random.shuffle(groups)
held_groups = set(groups[:2])
train = [r for r in records if r['group'] not in held_groups]
dev = [r for r in records if r['group'] in held_groups]
assert not {r['group'] for r in train} & {r['group'] for r in dev}
assert train and dev
(RUNS / 'week24_train.json').write_text(json.dumps(train, indent=2), encoding='utf-8')
(RUNS / 'week24_dev.json').write_text(json.dumps(dev, indent=2), encoding='utf-8')
print('Tiny demonstration split', len(train), len(dev), 'held groups', held_groups)
din = dout = 4096
r = 8
print('Dense params', din * dout, 'LoRA params', r * (din + dout), 'fraction', r * (din + dout) / (din * dout))
print('Visible synthetic records are not an independent benchmark or a useful-size SFT dataset.')
print('Example prompt and reviewed target:', json.dumps(records[0], indent=2))


# Audit targets and build a larger narrow-task dataset for the real project. Keep final task questions separate. Write why SFT might address the measured failure. Analytical adapter counts do not count as a training run; tutorial 26 performs one.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Tiny demonstration split 10 2 held groups {'retry', 'cache'}
# Dense params 16777216 LoRA params 65536 fraction 0.00390625
# Visible synthetic records are not an independent benchmark or a useful-size SFT dataset.
# Example prompt and reviewed target: {
#   "task_id": "q01",
#   "group": "timeout",
#   "version": "2",
#   "messages": [
#     {
#       "role": "user",
#       "content": "Use this evidence only:\n[timeout-v2] version=2 Atlas SDK version 2 uses timeout_ms. The default request timeout is 30000 milliseconds. Set timeout_ms when creating the Client.\n[retry-v2] version=2 Atlas SDK version 2 retries transient errors at most 3 times by default. For this example, the first retry waits 100 milliseconds, the next 200 milliseconds, then 400 milliseconds. There is no jitter in this teaching fixture.\n[auth-v2] version=2 Provide an API token in the ATLAS_TOKEN environment variable. Never place the token in documentation examples, source control or request logs. An expired token produces error AUTH_EXPIRED.\n[config-v2] version=2 A version 2 configuration has exactly two fields: timeout_ms and retries. timeout_ms must be an integer from 1 through 120000 inclusive. retries must be an integer from 0 through 5 inclusive. Extra fields are rejected.\n[cache-v2] version=2 Cache keys include tenant scope, requested version, corpus revision, prompt revision and model revision. Invalidate dependent cached responses after a source deletion. This fixture does not implement a distributed cache.\n[pages-v2] version=2 List endpoints return next_cursor when additional results exist. Supply the returned nex
# [report excerpt; inspect the saved full JSON for every record]
# o call any tool.\n[concurrency-v2] version=2 The example client allows at most 4 simultaneous requests by default. Additional work should wait in a bounded queue. This statement concerns request concurrency, not the retry count.\nQuestion: What is the default request timeout?"
#     },
#     {
#       "role": "assistant",
#       "content": "{\"answer\": \"The default request timeout is 30000 milliseconds.\", \"citations\": [\"timeout-v2\"]}"
#     }
#   ]
# }
# ```
