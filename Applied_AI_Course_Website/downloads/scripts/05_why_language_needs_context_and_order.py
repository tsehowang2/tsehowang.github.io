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
emb={'I':np.array([1.,0.]),'like':np.array([0.,1.]),'school':np.array([1.,1.])}
a=['I','like','school'];b=list(reversed(a))
def mean(words):return np.mean([emb[w] for w in words],axis=0)
def recur(words):
    h=np.zeros(2)
    for w in words:h=np.tanh(.5*h+emb[w])
    return h
assert np.allclose(mean(a),mean(b))
print('Same mean:',mean(a),mean(b),'Different recurrence:',recur(a),recur(b))
assert not np.allclose(recur(a),recur(b))

