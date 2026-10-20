# %%
from pathlib import Path
import sys,json,time,random,hashlib,math
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
candidates=[Path.cwd(),Path.cwd().parent,Path.cwd()/'llm_course',Path('/content/llm_course')]
COURSE=next((p for p in candidates if (p/'coursekit'/'core.py').exists()),None)
if COURSE is None:
    raise RuntimeError('Extract the course ZIP first; run from llm_course or its notebooks directory.')
if str(COURSE) not in sys.path: sys.path.insert(0,str(COURSE))
RUNS=COURSE/'runs'
RUNS.mkdir(exist_ok=True)
from coursekit.core import *
random.seed(7)
print('Course root:',COURSE)

# %%
alpha_bar=.64;x0=2.;epsilon=-1.
xt=math.sqrt(alpha_bar)*x0+math.sqrt(1-alpha_bar)*epsilon
assert abs(xt-1)<1e-12
print('Diffusion forward-noising scalar example',xt)
required=['lab01_baseline.json','lab03_config.json','lab04_model.json','lab06_retrieval.json',
          'lab08_to_grade.json','lab10_training.json','lab11_trace.json','lab12_agent.json',
          'lab13_manifest.json','lab13_latency.json','lab15_incident.json']
audit=[{'artifact':name,'exists':(RUNS/name).exists()} for name in required]
for row in audit: print(row)
print('Existence is only a first check: inspect contents, manual grades and actual model/GPU run evidence.')
(RUNS/'lab16_audit.json').write_text(json.dumps(audit,indent=2))

