# # Tutorial 27: Tools functions workflows and agents
# 
# **Prerequisite:** lecture 27. **Compute:** CPU; no downloads.
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


# First calculate the price of three two-dollar tickets using a deterministic function. A language model may propose the arguments, but code executes the arithmetic. The following controller demonstrates schema and scope checks before model planning.

import json
assert calculate('100+200+400') == 700
print(validate_config({'timeout_ms': 30000, 'retries': 3}))
for expression in ['1/0', "__import__('os')", '2**100']:
    try:
        calculate(expression)
    except ToolError as e:
        print('Rejected', expression, str(e))
    else:
        raise AssertionError('Unsafe expression accepted')

def planner(state):
    if state['step'] == 0:
        return {'type': 'tool', 'name': 'lookup', 'arguments': {'query': 'retry policy backoff', 'version': state['version']}}
    if state['step'] == 1:
        return {'type': 'tool', 'name': 'calculate', 'arguments': {'expression': '100+200+400'}}
    return {'type': 'final', 'answer': 'The fixture gives cumulative waiting time of 700 ms before attempt four.', 'citations': ['retry-v2']}
result = run_agent('What is cumulative retry wait before attempt four?', planner, max_steps=4)
assert result['status'] == 'completed'
print(json.dumps(result, indent=2))
(RUNS / 'week27_trace.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
loop = lambda state: {'type': 'tool', 'name': 'calculate', 'arguments': {'expression': '1+1'}}
assert run_agent('test', loop, max_steps=3)['status'] == 'budget_exhausted'
hostile = lambda state: {'type': 'tool', 'name': 'export_secrets', 'arguments': {}}
assert run_agent('test', hostile)['status'] == 'invalid_action'


# This deterministic planner hard-codes arithmetic for one fixture. It tests the controller, not model intelligence. Add task predicates and different valid/invalid config cases. The synchronous controller does not interrupt blocking planners; deployment needs real timeout/cancellation enforcement.

# ### Parallel calls and dependencies
# The two read-only fixtures can run together. Multiplication must wait for both results. A native tool-call response still needs argument checks, authorization, and error handling before execution. This fixture tests scheduling, not an LLM provider API.

# Deterministic read-only fixtures: no network access or side effects.
from concurrent.futures import ThreadPoolExecutor

def fare():
    return 2

def count():
    return 3
with ThreadPoolExecutor(max_workers=2) as pool:
    fare_future = pool.submit(fare)
    count_future = pool.submit(count)
    price, tickets = (fare_future.result(), count_future.result())
answer = price * tickets
assert answer == 6
print('Two independent reads, then dependent multiply:', answer)


# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# {'valid': True, 'errors': []}
# Rejected 1/0 arithmetic failure
# Rejected __import__('os') operator or value not allowed
# Rejected 2**100 operator or value not allowed
# {
#   "status": "completed",
#   "answer": "The fixture gives cumulative waiting time of 700 ms before attempt four.",
#   "citations": [
#     "retry-v2"
#   ],
#   "trace": [
#     {
#       "step": 0,
#       "name": "lookup",
#       "arguments": {
#         "query": "retry policy backoff",
#         "version": "2"
#       },
#       "result": [
#         {
#           "id": "retry-v2",
#           "title": "Retry policy and backoff",
#           "version": "2",
#           "source": "fixture://atlas/2/retries",
#           "text": "Atlas SDK version 2 retries transient errors at most 3 times by default. For this example, the first retry waits 100 milliseconds, the next 200 milliseconds, then 400 milliseconds. There is no jitter in this teaching fixture.",
#           "score": 5.041719922280393
#         },
#         {
#           "id": "concurrency-v2",
#           "title": "Concurrent requests",
#           "version": "2",
#           "source": "fixture://atlas/2/concurrency",
#           "text": "The example client allows at most 4 simultaneous requests by default. Additional work should wait in a bounded queue. This statement concerns request concurrency, not the retry count.",
#           "score": 1.3814225730487937
#         }
#       ]
#     },
#     {
#       "step": 1,
#       "name": "calculate",
#       "arguments": {
#         "expression": "100+200+400"
#       },
#       "result": {
#         "value": 700
#       }
#     }
#   ],
#   "grounding_verified": false
# }
# Two independent reads, then dependent multiply: 6
# ```
