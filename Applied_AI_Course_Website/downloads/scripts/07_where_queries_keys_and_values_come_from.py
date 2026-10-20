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
x=np.array([[1.,0.],[0.,1.],[1.,1.]])
Wq=np.array([[1.,0.],[0.,1.]])
Wk=np.array([[1.,1.],[0.,1.]])
Wv=np.array([[2.,0.],[0.,4.]])
Q,K,V=x@Wq,x@Wk,x@Wv
q=Q[2];scores=q@K.T
e=np.exp(scores-scores.max());weights=e/e.sum()
print('Q',Q,'K',K,'V',V,'scores',scores,'weights',weights,'output',weights@V,sep='\n')
assert scores.shape==(3,);assert (weights@V).shape==(2,)

