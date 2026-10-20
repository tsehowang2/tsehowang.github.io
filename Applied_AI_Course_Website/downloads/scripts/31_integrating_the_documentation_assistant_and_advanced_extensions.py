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
required={'corpus_revision':None,'embedding_revision':None,'index_revision':None,'generator_revision':None,'prompt_revision':None,'tool_revision':None,'evaluation_run':None}
draft=RUNS/'week31_release_draft.json'
if not draft.exists():draft.write_text(json.dumps(required,indent=2))
manifest=json.loads(draft.read_text());missing=[k for k,v in manifest.items() if not v]
print('Unfilled fields:',missing)
print('Draft only. Supply actual identities; a complete manifest is necessary but not sufficient for release.')

