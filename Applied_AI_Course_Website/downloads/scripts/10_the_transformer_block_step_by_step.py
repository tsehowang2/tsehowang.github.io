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
rng=np.random.default_rng(7);x=rng.normal(size=(6,4))
def ln(a):return (a-a.mean(-1,keepdims=True))/np.sqrt(a.var(-1,keepdims=True)+1e-5)
h=ln(x);print('Row means',h.mean(-1),'row variances',h.var(-1))
W1=rng.normal(size=(4,8));W2=rng.normal(size=(8,4))
mlp=np.maximum(h@W1,0)@W2
y=x+mlp
assert y.shape==x.shape
changed=h.copy();changed[2]+=1
delta=np.maximum(changed@W1,0)@W2-mlp
assert np.allclose(np.delete(delta,2,axis=0),0)
print('MLP is positionwise; a changed row did not change the other rows.')

