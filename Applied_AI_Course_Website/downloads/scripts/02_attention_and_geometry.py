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
import numpy as np
q=np.array([1.,0.]);a=np.array([.8,.6]);b=np.array([4.,3.])
for v in [a,b]: print('dot',q@v,'cosine',q@v/(np.linalg.norm(q)*np.linalg.norm(v)))
def softmax(z):
    e=np.exp(z-z.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)
keys=np.array([[1.,0.],[0.,1.]]);values=np.array([[2.,0.],[0.,4.]])
weights=softmax((q@keys.T/np.sqrt(2))[None,:]);out=weights@values
print('weights',weights,'output',out)
assert np.allclose(out,[[1.3395231,1.3209538]],atol=1e-5)

# %%
def attention(x,wq,wk,wv,causal=True):
    Q,K,V=x@wq,x@wk,x@wv
    scores=Q@K.T/np.sqrt(Q.shape[-1])
    if causal: scores=np.where(np.triu(np.ones(scores.shape,dtype=bool),k=1),-np.inf,scores)
    weights=softmax(scores)
    return weights@V,weights
rng=np.random.default_rng(7)
x=rng.normal(size=(6,4));mats=[rng.normal(size=(4,3)) for _ in range(3)]
changed=x.copy();changed[-1]+=10
original,_=attention(x,*mats);modified,w=attention(changed,*mats)
assert np.allclose(original[:-1],modified[:-1])
unmasked,_=attention(x,*mats,causal=False);changed_unmasked,_=attention(changed,*mats,causal=False)
assert not np.allclose(unmasked[:-1],changed_unmasked[:-1])
assert np.allclose(w.sum(axis=-1),1)
print('Causal invariance passes; unmasked operation fails the same property as expected.')
for T in [128,512,2048]: print('T',T,'float32 score MiB per head',T*T*4/2**20)

