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
def softmax(z):
    z=np.asarray(z,dtype=float);e=np.exp(z-z.max());return e/e.sum()
z=np.array([2.,1.,0.]);target=1
p=softmax(z);loss=-np.log(p[target]);gradient=p.copy();gradient[target]-=1
print('bus/train/bicycle probabilities',p,'train loss',loss,'gradient',gradient)
eps=1e-5
numerical=[]
for i in range(3):
    d=np.zeros(3);d[i]=eps
    numerical.append((-np.log(softmax(z+d)[target])+np.log(softmax(z-d)[target]))/(2*eps))
assert np.allclose(gradient,numerical,atol=1e-7)
assert np.allclose(softmax(z+10000),p)
print('After a small gradient step:',softmax(z-.1*gradient))

