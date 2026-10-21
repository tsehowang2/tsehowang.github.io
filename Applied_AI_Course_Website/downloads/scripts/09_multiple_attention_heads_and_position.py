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


import numpy as np
B,T,d_model,H=1,6,8,2;d_head=d_model//H
x=np.arange(B*T*d_model).reshape(B,T,d_model)
heads=x.reshape(B,T,H,d_head).transpose(0,2,1,3)
merged=heads.transpose(0,2,1,3).reshape(B,T,d_model)
print('Original',x.shape,'split',heads.shape,'merged',merged.shape)
assert np.array_equal(x,merged)
pos=np.arange(T)[:,None];dims=np.arange(0,d_model,2)
angles=pos/(10000**(dims/d_model))
pe=np.zeros((T,d_model));pe[:,0::2]=np.sin(angles);pe[:,1::2]=np.cos(angles)
print('Position vectors:',pe[:3])
assert not np.allclose(pe[0],pe[1])


import math
def rotate(v,theta):
    c,s=math.cos(theta),math.sin(theta);return [v[0]*c-v[1]*s,v[0]*s+v[1]*c]
def dp(a,b):return sum(x*y for x,y in zip(a,b))
q=k=[1.,0.]
for q_deg,k_deg in [(0,0),(30,90),(60,120)]:
    value=dp(rotate(q,math.radians(q_deg)),rotate(k,math.radians(k_deg)))
    expected=math.cos(math.radians(k_deg-q_deg))
    assert math.isclose(value,expected,abs_tol=1e-12)
    print(q_deg,k_deg,'relative offset',k_deg-q_deg,'dot',value)
