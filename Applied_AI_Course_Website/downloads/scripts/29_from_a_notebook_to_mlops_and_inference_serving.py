# # Tutorial 29: From a notebook to MLOps and inference serving
# 
# **Prerequisite:** lecture 29. **Compute:** CPU; no downloads.
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


# A local notebook is a process you operate manually. A service receives requests, checks them and returns responses under a recorded configuration. The extraction API below teaches that boundary; it is not a production LLM server.

import json
manifest = make_manifest({'prompt_revision': 'baseline-extractor-v1', 'tool_revision': 'strict-tools-v1'})
assert verify_manifest(manifest)
(RUNS / 'week29_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
bad = dict(manifest)
bad['corpus_sha256'] = 'changed'
try:
    verify_manifest(bad)
except ValueError as e:
    print('Rejected incompatible manifest:', e)


# The next cell starts a loopback extraction API as a subprocess and stops it afterward. It is not an LLM inference benchmark. The OS selects a free loopback port; server errors are preserved in week29_server.stderr.log.

import sys
import json
import math
import time
import subprocess, urllib.request, urllib.error
import queue, threading
stderr_log = open(RUNS / 'week29_server.stderr.log', 'w', encoding='utf-8')
process = subprocess.Popen([sys.executable, '-u', '-m', 'coursekit.server', '--port', '0'], cwd=COURSE, stdout=subprocess.PIPE, stderr=stderr_log, text=True, encoding='utf-8')
line_queue = queue.Queue()
threading.Thread(target=lambda: line_queue.put(process.stdout.readline()), daemon=True).start()
try:
    announcement = json.loads(line_queue.get(timeout=10))
    url = 'http://127.0.0.1:' + str(announcement['port'])
    with urllib.request.urlopen(url + '/health', timeout=3) as response:
        assert json.load(response)['status'] == 'ok'
    payload = json.dumps({'question': 'default request timeout', 'version': '2'}).encode()
    timings = []
    for _ in range(200):
        start = time.perf_counter()
        request = urllib.request.Request(url + '/ask', data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.load(response)
        timings.append((time.perf_counter() - start) * 1000)
    assert 'timeout-v2' in result['citations'] and 'timeout-v1' not in result['citations']
    ordered = sorted(timings)
    print('Local extractor median ms', ordered[len(ordered) // 2], 'descriptive sample p95 (200 sequential local requests; not production tail latency)', ordered[math.ceil(0.95 * len(ordered)) - 1])
    (RUNS / 'week29_latency.json').write_text(json.dumps({'milliseconds': timings, 'result': result, 'service': 'local extractor'}, indent=2), encoding='utf-8')
finally:
    process.terminate()
    process.wait(timeout=5)
    process.stdout.close()
    stderr_log.close()


# ### Continuous batching admission trace
# B finishes after one decode step and C enters at the next step while A continues. This is a scheduling illustration; it does not implement PagedAttention or measure a serving runtime.

requests = {'A': 3, 'B': 1}
completed = []
trace = []
for step in range(1, 4):
    if step == 2:
        requests['C'] = 2
    active = list(requests)
    for request in active:
        requests[request] -= 1
        if requests[request] == 0:
            completed.append(request)
    trace.append((step, active, completed.copy()))
    requests = {key: value for key, value in requests.items() if value > 0}
assert trace[1][1] == ['A', 'C'] and set(completed) == {'A', 'B', 'C'}
print('Decode-step scheduling fixture:', trace)


# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Rejected incompatible manifest: corpus hash mismatch; rebuild/re-evaluate dependent artifacts
# Local extractor median ms 5.970099940896034 descriptive sample p95 (200 sequential local requests; not production tail latency) 19.920799881219864
# Decode-step scheduling fixture: [(1, ['A', 'B'], ['B']), (2, ['A', 'C'], ['B']), (3, ['A', 'C'], ['B', 'A', 'C'])]
# ```
