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
import numpy as np
rng=np.random.default_rng(7);din,dout,rank=4,3,2
W=rng.normal(size=(dout,din));A=rng.normal(size=(rank,din));B=np.zeros((dout,rank))
x=rng.normal(size=din);scale=1.
assert np.allclose(W@x,(W+scale*B@A)@x)
B=rng.normal(size=(dout,rank))
assert np.allclose((W+scale*B@A)@x,W@x+scale*B@(A@x))
print('Dense parameters',W.size,'adapter parameters',A.size+B.size,'delta rank',np.linalg.matrix_rank(B@A))
for r in [1,2,4,16]:print('2048x4096 rank',r,'adapter count',r*(2048+4096))

