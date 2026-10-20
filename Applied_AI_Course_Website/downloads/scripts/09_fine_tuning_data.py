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
docs=load_docs();byid={d['id']:d for d in docs};records=[]
for t in json.loads((COURSE/'data/queries.json').read_text()):
    if t['version']=='all': continue
    context='\n'.join('['+i+'] '+byid[i]['text'] for i in t['relevant'])
    target={'answer':' '.join(byid[i]['text'] for i in t['relevant']) if t['relevant'] else 'Insufficient evidence.',
            'citations':t['relevant']}
    records.append({'group':t['group'],'messages':[
        {'role':'user','content':'Use this evidence only:\n'+context+'\nQuestion: '+t['question']},
        {'role':'assistant','content':json.dumps(target)}]})
groups=sorted({r['group'] for r in records});random.shuffle(groups);held_groups=set(groups[:2])
train=[r for r in records if r['group'] not in held_groups];dev=[r for r in records if r['group'] in held_groups]
assert not ({r['group'] for r in train}&{r['group'] for r in dev})
assert train and dev
(RUNS/'lab09_train.json').write_text(json.dumps(train,indent=2))
(RUNS/'lab09_dev.json').write_text(json.dumps(dev,indent=2))
print('Tiny demonstration split',len(train),len(dev),'held groups',held_groups)
din=dout=4096;r=8
print('Dense params',din*dout,'LoRA params',r*(din+dout),'fraction',r*(din+dout)/(din*dout))
print('Visible synthetic records are not an independent benchmark or a useful-size SFT dataset.')

