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
# Synthetic telemetry, not measured service traffic.
events=[{'id':i,'empty':i<3,'ms':10+i%5,'release':'A'} for i in range(100)]
events += [{'id':100+i,'empty':i<25,'ms':10+i%5,'release':'B'} for i in range(100)]
for release in ['A','B']:
    rows=[r for r in events if r['release']==release]
    rate=sum(r['empty'] for r in rows)/len(rows)
    print('Synthetic release',release,'n',len(rows),'empty rate',rate,'alert',len(rows)>=50 and rate>.2)
attacks=[
 {'type':'tool','name':'export_secrets','arguments':{}},
 {'type':'tool','name':'calculate','arguments':{'expression':"__import__('os')"}},
 {'type':'tool','name':'lookup','arguments':{'query':'timeout','version':'1'}},
 {'type':'tool','name':'validate_config','arguments':{'config':{'timeout_ms':True,'retries':1}}}
]
for proposal in attacks:
    result=run_agent('test',lambda state,p=proposal:p,version='2',max_steps=2)
    print(proposal['name'],result['status'])
    if proposal['name']!='validate_config': assert result['status']=='invalid_action'
    else: assert not result['trace'][0]['result']['valid']
print('A valid tool call returning invalid config differs from an invalid call schema.')

# %%
import subprocess
subprocess.check_call([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=COURSE)
previous=make_manifest({'release':'qualifying-A'})
candidate=dict(previous);candidate['release']='candidate-B';candidate['corpus_sha256']='incompatible'
try:
    verify_manifest(candidate);decision='eligible_for_further_quality_checks'
except ValueError: decision='reject_and_restore_previous'
assert decision=='reject_and_restore_previous' and verify_manifest(previous)
print(decision)
for scalable in [.7,.8]: print('Ideal 4-way speedup',scalable,1/((1-scalable)+scalable/4))
report={'trigger':'incompatible corpus hash','impact':'candidate refused before release',
        'mitigation':'restore prior manifest','regression':'manifest rejects changed corpus hash',
        'remaining_checks':['semantic quality','real index/model compatibility','load behavior']}
(RUNS/'week30_incident.json').write_text(json.dumps(report,indent=2))

