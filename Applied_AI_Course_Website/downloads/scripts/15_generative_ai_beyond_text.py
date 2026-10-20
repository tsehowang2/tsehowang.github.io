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
rng=np.random.default_rng(7);x0=np.zeros((8,8));x0[2:6,3:5]=1
noise=rng.normal(size=x0.shape)
for alpha_bar in [.99,.64,.1]:
    xt=np.sqrt(alpha_bar)*x0+np.sqrt(1-alpha_bar)*noise
    recovered=(xt-np.sqrt(1-alpha_bar)*noise)/np.sqrt(alpha_bar)
    assert np.allclose(recovered,x0)
    print('alpha_bar',alpha_bar,'noisy center',np.round(xt[2:6,2:6],2))

