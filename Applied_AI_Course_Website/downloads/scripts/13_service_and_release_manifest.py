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
manifest=make_manifest({'prompt_revision':'baseline-extractor-v1','tool_revision':'strict-tools-v1'})
assert verify_manifest(manifest)
(RUNS/'lab13_manifest.json').write_text(json.dumps(manifest,indent=2))
bad=dict(manifest);bad['corpus_sha256']='changed'
try: verify_manifest(bad)
except ValueError as e: print('Rejected incompatible manifest:',e)

# %%
import subprocess,urllib.request,urllib.error
process=subprocess.Popen([sys.executable,'-m','coursekit.server'],cwd=COURSE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    for attempt in range(30):
        if process.poll() is not None: raise RuntimeError('Server exited; check whether port 8000 is occupied.')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=1) as response:
                assert json.load(response)['status']=='ok'
            break
        except (OSError,AssertionError): time.sleep(.1)
    else: raise RuntimeError('Server did not become healthy')
    payload=json.dumps({'question':'default request timeout','version':'2'}).encode()
    timings=[]
    for _ in range(20):
        start=time.perf_counter()
        request=urllib.request.Request('http://127.0.0.1:8000/ask',data=payload,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=3) as response: result=json.load(response)
        timings.append((time.perf_counter()-start)*1000)
    assert 'timeout-v2' in result['citations'] and 'timeout-v1' not in result['citations']
    ordered=sorted(timings)
    print('Local extractor median ms',ordered[len(ordered)//2],'nearest-rank p95',ordered[math.ceil(.95*len(ordered))-1])
    (RUNS/'lab13_latency.json').write_text(json.dumps({'milliseconds':timings,'result':result,'service':'local extractor'},indent=2))
finally:
    process.terminate();process.wait(timeout=5)

