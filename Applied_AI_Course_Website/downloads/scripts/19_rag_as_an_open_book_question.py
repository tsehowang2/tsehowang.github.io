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
handbook=[{'id':'bus','text':'The school bus leaves at 8:00.'},{'id':'lunch','text':'Lunch starts at 12:00.'}]
question='When does the school bus leave?'
evidence=[d for d in handbook if 'bus' in d['text'].lower()]
prompt='Use only the evidence. Cite IDs. If missing, say insufficient evidence.\n'
prompt+='\n'.join('['+d['id']+'] '+d['text'] for d in evidence)
prompt+='\nQuestion: '+question
print(prompt)
assert evidence[0]['id']=='bus'
index=LexicalIndex(load_docs())
print('Deterministic extractor, not an LLM:',extract_answer('default request timeout',index,'2'))

