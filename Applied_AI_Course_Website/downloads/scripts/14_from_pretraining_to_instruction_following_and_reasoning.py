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
prompt='User How do I go to school Assistant'
answer='Take the bus'
words=(prompt+' '+answer).split();vocab={w:i for i,w in enumerate(dict.fromkeys(words))}
ids=[vocab[w] for w in words];n=len(prompt.split())
labels=[-100]*n+ids[n:]
print(list(zip(words,ids,labels)))
assert all(x==-100 for x in labels[:n])
print('At causal logit position',n-1,'the next-token target is',words[n])

