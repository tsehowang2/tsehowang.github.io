# %%
from pathlib import Path
import sys,json,time,random,hashlib,math
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
candidates=[Path.cwd(),Path.cwd().parent,Path.cwd()/'llm_course',Path.cwd()/'llm_course_v2',Path('/content/llm_course'),Path('/content/llm_course_v2')]
COURSE=next((p for p in candidates if (p/'coursekit/core.py').exists()),None)
if COURSE is None: raise RuntimeError('Extract the complete course pack; run from its root or notebooks directory.')
if str(COURSE) not in sys.path:sys.path.insert(0,str(COURSE))
RUNS=COURSE/'runs';RUNS.mkdir(exist_ok=True)
from coursekit.core import *
random.seed(7)
print('Course root:',COURSE)

# %%
required=['week12_config.json','week18_retrieval.json',
          'week21_to_grade.json','week26_training.json','week27_trace.json','week28_agent.json',
          'week29_manifest.json','week29_latency.json','week30_incident.json']
audit=[{'artifact':name,'exists':(RUNS/name).exists()} for name in required]
for row in audit: print(row)
print('Existence is only a first check: inspect contents, manual grades and actual model/GPU run evidence.')
(RUNS/'week32_audit.json').write_text(json.dumps(audit,indent=2))

