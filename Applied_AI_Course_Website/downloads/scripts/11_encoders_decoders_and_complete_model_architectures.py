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
T,S,D=6,4,8
encoder_mask=np.ones((S,S),dtype=bool)
decoder_mask=np.tril(np.ones((T,T),dtype=bool))
cross_mask=np.ones((T,S),dtype=bool)
print('Encoder visibility:',encoder_mask.astype(int),'Decoder visibility:',decoder_mask.astype(int),'Cross visibility:',cross_mask.astype(int),sep='\n')
q=np.zeros((T,D));k=np.zeros((S,D));v=np.ones((S,D))
assert (q@k.T).shape==(T,S)
assert ((np.ones((T,S))/S)@v).shape==(T,D)

