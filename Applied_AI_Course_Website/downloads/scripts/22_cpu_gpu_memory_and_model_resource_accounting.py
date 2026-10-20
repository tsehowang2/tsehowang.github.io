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
parameters=300_000_000
for bytes_per_parameter in [2,4,16]:
    size=parameters*bytes_per_parameter
    print(bytes_per_parameter,'bytes/parameter:',size/1e9,'GB',size/2**30,'GiB')
layers,batch,tokens,kv_heads,head_dim,bytes_per_element=24,1,2048,8,64,2
cache=2*layers*batch*tokens*kv_heads*head_dim*bytes_per_element
assert cache/2**20==96
print('Hypothetical KV cache:',cache/2**20,'MiB')

