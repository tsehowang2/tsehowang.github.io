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
assert calculate('100+200+400')==700
print(validate_config({'timeout_ms':30000,'retries':3}))
for expression in ['1/0',"__import__('os')",'2**100']:
    try: calculate(expression)
    except ToolError as e: print('Rejected',expression,str(e))
def planner(state):
    if state['step']==0:
        return {'type':'tool','name':'lookup','arguments':{'query':'retry policy backoff','version':state['version']}}
    if state['step']==1:
        return {'type':'tool','name':'calculate','arguments':{'expression':'100+200+400'}}
    return {'type':'final','answer':'The fixture gives cumulative waiting time of 700 ms before attempt four.','citations':['retry-v2']}
result=run_agent('What is cumulative retry wait before attempt four?',planner,max_steps=4)
assert result['status']=='completed'
print(json.dumps(result,indent=2));(RUNS/'week27_trace.json').write_text(json.dumps(result,indent=2))
loop=lambda state:{'type':'tool','name':'calculate','arguments':{'expression':'1+1'}}
assert run_agent('test',loop,max_steps=3)['status']=='budget_exhausted'
hostile=lambda state:{'type':'tool','name':'export_secrets','arguments':{}}
assert run_agent('test',hostile)['status']=='invalid_action'

