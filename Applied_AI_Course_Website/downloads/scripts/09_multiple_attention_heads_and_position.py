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
B,T,D,H=1,6,8,2;dh=D//H
x=np.arange(B*T*D).reshape(B,T,D)
heads=x.reshape(B,T,H,dh).transpose(0,2,1,3)
merged=heads.transpose(0,2,1,3).reshape(B,T,D)
print('Original',x.shape,'split',heads.shape,'merged',merged.shape)
assert np.array_equal(x,merged)
pos=np.arange(T)[:,None];dims=np.arange(0,D,2)
angles=pos/(10000**(dims/D))
pe=np.zeros((T,D));pe[:,0::2]=np.sin(angles);pe[:,1::2]=np.cos(angles)
print('Position vectors:',pe[:3])
assert not np.allclose(pe[0],pe[1])

