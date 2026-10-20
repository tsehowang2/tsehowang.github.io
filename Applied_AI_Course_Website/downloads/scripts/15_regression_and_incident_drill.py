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
import subprocess
subprocess.check_call([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=COURSE)
previous=make_manifest({'release':'qualifying-A'})
candidate=dict(previous);candidate['release']='candidate-B';candidate['corpus_sha256']='incompatible'
try:
    verify_manifest(candidate);decision='eligible_for_further_quality_checks'
except ValueError: decision='reject_and_restore_previous'
assert decision=='reject_and_restore_previous' and verify_manifest(previous)
print(decision)
for scalable in [.7,.8]: print('Ideal 4-way speedup',scalable,1/((1-scalable)+scalable/4))
report={'trigger':'incompatible corpus hash','impact':'candidate refused before release',
        'mitigation':'restore prior manifest','regression':'manifest rejects changed corpus hash',
        'remaining_checks':['semantic quality','real index/model compatibility','load behavior']}
(RUNS/'lab15_incident.json').write_text(json.dumps(report,indent=2))

