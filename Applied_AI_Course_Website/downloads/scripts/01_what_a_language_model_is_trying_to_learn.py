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
from collections import Counter
sentences=['I go to school by bus']*3+['I go to school by train','I go to school by bicycle']
prefix='I go to school by '
counts=Counter(s[len(prefix):] for s in sentences if s.startswith(prefix))
probs={word:count/sum(counts.values()) for word,count in counts.items()}
print(counts,probs)
assert probs=={'bus':.6,'train':.2,'bicycle':.2}

# %%
counts['train']+=2
print({w:c/sum(counts.values()) for w,c in counts.items()})
assert sum(counts.values())==7
print('Three-token chain probability:',.5*.4*.25)

