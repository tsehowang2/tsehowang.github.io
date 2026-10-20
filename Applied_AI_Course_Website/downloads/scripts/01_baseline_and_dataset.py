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
docs=load_docs();tasks=json.loads((COURSE/'data/queries.json').read_text())
index=LexicalIndex(docs)
for d in docs[:2]: print(d)
records=[]
for t in tasks:
    response=extract_answer(t['question'],index,t['version'])
    metric=retrieval_metrics(response['citations'],t['relevant'],3)
    records.append({'id':t['id'],'question':t['question'],'response':response,'retrieval':metric})
    print(t['id'],response['citations'],metric)
(RUNS/'lab01_baseline.json').write_text(json.dumps(records,indent=2))
answerable=[r for r in records if r['retrieval']['recall'] is not None]
print('Mean answerable Recall@3:',sum(r['retrieval']['recall'] for r in answerable)/len(answerable))
print('Unanswerable cases need separate manual refusal/relevance scoring.')

# %%
for text in ['request timeout','timeout_ms=30000','你好','AUTH_EXPIRED']:
    print(repr(text),'characters',len(text),'UTF8 bytes',len(text.encode('utf-8')))
remaining=2048-300-550
assert remaining==1198
print('Evidence budget before extra template overhead:',remaining)

