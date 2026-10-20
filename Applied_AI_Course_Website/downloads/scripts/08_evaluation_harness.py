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
tasks=json.loads((COURSE/'data/queries.json').read_text());index=LexicalIndex(load_docs())
records=[]
for t in tasks:
    result=extract_answer(t['question'],index,t['version'])
    records.append({'task_id':t['id'],'group':t['group'],'expected':t['relevant'],'result':result,
                    'retrieval':retrieval_metrics(result['citations'],t['relevant'],3),
                    'manual_correct':None,'manual_supported':None,'manual_refusal_appropriate':None})
(RUNS/'lab08_to_grade.json').write_text(json.dumps(records,indent=2))
assert retrieval_metrics(['C','A','D'],['A','B'],3)=={'recall':.5,'hit':1,'rr':.5}
print('Illustrative paired data, NOT model results:',paired_bootstrap([1,1,0,0,1,0],[1,1,1,0,1,0]))
print('Fill manual fields; preserve failures and evaluate unsupported cases separately.')

