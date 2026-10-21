# # Tutorial 32: Review defense and a continuing learning plan
# 
# **Prerequisite:** lecture 32. **Compute:** CPU audit; completed model/GPU artifacts from previous labs.
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


# Use this week to reproduce and defend, not add features. The audit checks artifact presence; your report must supply the actual task results, failure analysis and release reasoning.

import json
from coursekit.release_audit import audit_release
report = audit_release(COURSE, RUNS)
print(json.dumps(report, indent=2))
(RUNS / 'week32_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
assert not report['ready'] or not report['missing_artifacts']
print('A draft or missing artifact is reported as incomplete; inspect measured evidence before release.')


# Complete the release checklist from a clean directory, not just this existence audit. Demonstrate supported/absent/version/tool/failure cases. Optional multimodal extension: label table cells/header associations and assess extraction separately. Take semester exam 2, prepare the defense and state incomplete tasks explicitly.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# {
#   "missing_artifacts": [
#     "week12_config.json",
#     "tiny_lm.pt",
#     "week18_retrieval.json",
#     "week26_training.json",
#     "week26_adapter",
#     "week28_agent.json",
#     "project_cases.json",
#     "project_results.json",
#     "manual_grades.json",
#     "environment.txt",
#     "release_manifest.json",
#     "limitations.md"
#   ],
#   "invalid_artifacts": [
#     "project_cases.json: need at least 40 manually checked cases",
#     "release_manifest.json: unfilled corpus_revision, chunker_revision, embedding_revision, index_revision, generator_revision, tokenizer_revision, adapter_revision, prompt_revision, tool_revision, environment_revision, evaluation_run"
#   ],
#   "checked_cases": 0,
#   "ready": false,
#   "scope": "Artifact checks only. Independently review quality, support, privacy and measurements."
# }
# A draft or missing artifact is reported as incomplete; inspect measured evidence before release.
# ```
