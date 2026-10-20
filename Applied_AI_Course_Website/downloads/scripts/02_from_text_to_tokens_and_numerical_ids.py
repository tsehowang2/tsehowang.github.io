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
sentence='I go to school by bus'
words=sentence.split();vocab={w:i for i,w in enumerate(words)}
ids=[vocab[w] for w in words]
print(vocab,ids)
inputs,targets=ids[:-1],ids[1:]
print('Inputs:',inputs,'Targets:',targets)
assert inputs[1:]==targets[:-1]
print('Characters:',len(sentence),'UTF-8 bytes:',len(sentence.encode()))

