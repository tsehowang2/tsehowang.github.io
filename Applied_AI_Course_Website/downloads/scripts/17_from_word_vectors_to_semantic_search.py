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


texts=['The school bus leaves at eight.','Lunch starts at noon.','The library closes at five.']
query='When can I catch the bus?'
print('Query:',query)
for i,t in enumerate(texts):print(i,t)
import numpy as np
# Invented normalized geometry to inspect exact search, NOT model embeddings.
vectors=np.array([[1.,0.],[0.,1.],[.6,.8]])
q=np.array([1.,0.]);scores=vectors@q
print('Illustrative ranking:',np.argsort(-scores).tolist())


import math
scores = [[2., 1., 0.], [0., 2., 1.], [1., 0., 2.]]
def candidate_probs(row):
    exps = [math.exp(x - max(row)) for x in row]
    return [x / sum(exps) for x in exps]
probs = [candidate_probs(row) for row in scores]
losses = [-math.log(row[i]) for i, row in enumerate(probs)]
print("Candidate probabilities:", probs)
print("Mean contrastive loss:", sum(losses) / len(losses))
print("False-negative example Q1 loss:", -math.log(candidate_probs([2., 2., 0.])[0]))


RUN_EMBEDDINGS=False
if RUN_EMBEDDINGS:
    import subprocess
    subprocess.check_call([sys.executable,'-m','pip','install','sentence-transformers>=3,<6'])
    from sentence_transformers import SentenceTransformer
    encoder=SentenceTransformer('intfloat/multilingual-e5-small')
    matrix=encoder.encode(['passage: '+t for t in texts],normalize_embeddings=True)
    q=encoder.encode(['query: '+query],normalize_embeddings=True)[0]
    print('Actual model scores:',matrix@q)
    print('Actual ranking:',np.argsort(-(matrix@q)).tolist())
else:print('Actual embedding model skipped; enable RUN_EMBEDDINGS for model results.')


import math
cosines=[.8,.6,.2]
def probs_at_temperature(xs,tau):
    logits=[x/tau for x in xs];m=max(logits);e=[math.exp(x-m) for x in logits]
    return [x/sum(e) for x in e]
for tau in [1.,.05]:
    p=probs_at_temperature(cosines,tau);print('tau',tau,'p',p,'positive loss',-math.log(p[0]))
assert -math.log(probs_at_temperature(cosines,.05)[0]) < -math.log(probs_at_temperature(cosines,1.)[0])
