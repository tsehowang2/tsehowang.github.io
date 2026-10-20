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
texts=['The school bus leaves at eight.','Lunch starts at noon.','The library closes at five.']
query='When can I catch the bus?'
print('Query:',query)
for i,t in enumerate(texts):print(i,t)
import numpy as np
# Invented normalized geometry to inspect exact search, NOT model embeddings.
vectors=np.array([[1.,0.],[0.,1.],[.6,.8]])
q=np.array([1.,0.]);scores=vectors@q
print('Illustrative ranking:',np.argsort(-scores).tolist())

# %%
RUN_EMBEDDINGS=False
if RUN_EMBEDDINGS:
    import subprocess
    subprocess.check_call([sys.executable,'-m','pip','install','sentence-transformers>=3,<6'])
    from sentence_transformers import SentenceTransformer
    encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    matrix=encoder.encode(texts,normalize_embeddings=True)
    q=encoder.encode([query],normalize_embeddings=True)[0]
    print('Actual model scores:',matrix@q)
    print('Actual ranking:',np.argsort(-(matrix@q)).tolist())
else:print('Actual embedding model skipped; enable RUN_EMBEDDINGS for model results.')

