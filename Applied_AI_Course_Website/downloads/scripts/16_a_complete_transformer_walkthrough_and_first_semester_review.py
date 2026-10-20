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
B,T,D,H,V=2,6,8,2,20
dh=D//H
trace={'token IDs':(B,T),'embeddings':(B,T,D),'split Q/K/V':(B,H,T,dh),'scores':(B,H,T,T),'merged attention':(B,T,D),'vocabulary logits':(B,T,V),'shifted targets':(B,T)}
for name,shape in trace.items():print(name,shape)
score_bytes=B*H*T*T*4
assert score_bytes==576
print('Explicit float32 scores in this example:',score_bytes,'bytes')
print('Use the week-12 implementation to verify these shapes on a live model.')

