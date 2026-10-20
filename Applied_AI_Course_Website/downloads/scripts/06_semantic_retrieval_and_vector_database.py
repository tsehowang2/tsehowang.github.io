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
import subprocess
subprocess.check_call([sys.executable,'-m','pip','install','-q','sentence-transformers>=3,<6','qdrant-client>=1.12,<2','transformers>=4.45,<5'])
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient,models
docs=load_docs();tasks=json.loads((COURSE/'data/queries.json').read_text())
EMBED='sentence-transformers/all-MiniLM-L6-v2'
encoder=SentenceTransformer(EMBED,device='cpu')
vectors=encoder.encode([d['title']+'\n'+d['text'] for d in docs],normalize_embeddings=True)
assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4)
print('Vector shape',vectors.shape,'model max input',encoder.max_seq_length)
path=RUNS/'lab06_qdrant';client=QdrantClient(path=str(path));collection='atlas_minilm'
if client.collection_exists(collection): client.delete_collection(collection)
client.create_collection(collection,vectors_config=models.VectorParams(size=vectors.shape[1],distance=models.Distance.COSINE))
client.upsert(collection,points=[models.PointStruct(id=i,vector=vectors[i].tolist(),payload=d) for i,d in enumerate(docs)])
def semantic_search(question,version='2',k=3):
    q=encoder.encode(question,normalize_embeddings=True).tolist()
    filt=None if version=='all' else models.Filter(must=[models.FieldCondition(key='version',match=models.MatchValue(value=version))])
    return client.query_points(collection_name=collection,query=q,query_filter=filt,limit=k).points
rows=[];lexical=LexicalIndex(docs)
for task in tasks:
    hits=semantic_search(task['question'],task['version']);ids=[p.payload['id'] for p in hits]
    row={'id':task['id'],'dense':retrieval_metrics(ids,task['relevant'],3),
         'lexical':retrieval_metrics([d['id'] for d in lexical.search(task['question'],task['version'])],task['relevant'],3),
         'dense_ids':ids}
    rows.append(row);print(row)
client.close();client=QdrantClient(path=str(path))
assert all(p.payload['version']=='2' for p in semantic_search('default timeout','2'))
print('Persist/reopen/filter check passed');client.close()
(RUNS/'lab06_retrieval.json').write_text(json.dumps(rows,indent=2))

