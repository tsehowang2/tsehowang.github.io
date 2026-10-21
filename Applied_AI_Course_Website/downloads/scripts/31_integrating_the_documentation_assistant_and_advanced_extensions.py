# # Tutorial 31: Integrating the documentation assistant and advanced extensions
# 
# **Prerequisite:** lecture 31. **Compute:** CPU audit; existing model artifacts for integrated tests.
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
# For each component, identify the actual artifact and which evaluation depends on it.

# Integrate only components with evidence. This audit records release dependencies; it cannot prove quality from file existence. Bring the per-task measurements and compare candidate and baseline under the same protocol.

import json
import sys
from coursekit.release_audit import audit_release
report = audit_release(COURSE, RUNS)
print(json.dumps(report, indent=2))
(RUNS / 'week31_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


# ### Independent experiment
# Run supported, absent, wrong-version, executable-tool and failure demonstrations. Present a paired evaluation table with real task IDs. Remove components that add cost without reliable benefit. Prepare the defense rather than adding an unrelated framework.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

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
# ```
