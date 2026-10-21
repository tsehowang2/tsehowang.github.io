# # Tutorial 18: A vector database from a list of vectors
# 
# **Prerequisite:** lecture 18. **Compute:** CPU with embedding download and Qdrant client.
# 
# Predict the result before each code cell. Run cells in order from a fresh kernel. Tutorials 01–16 run without coursekit. Tutorials 17–32 use the extracted full course pack linked on the tutorial page. Reference outputs are labeled below; keep your own actual results.

import random
from pathlib import Path
import sys, json, time, random, hashlib, math
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
candidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / 'llm_course', Path.cwd() / 'llm_course_v2', Path('/content/llm_course'), Path('/content/llm_course_v2')]
COURSE = next((p for p in candidates if (p / 'coursekit/core.py').exists()), None)
if COURSE is None:
    raise RuntimeError('Extract the complete course pack; run from its root or notebooks directory.')
if str(COURSE) not in sys.path:
    sys.path.insert(0, str(COURSE))
RUNS = COURSE / 'runs'
RUNS.mkdir(exist_ok=True)
from coursekit.core import *
random.seed(7)
print('Course root:', COURSE)


# ### Current embedding recipe
# Use intfloat/multilingual-e5-small with query: before every query and passage: before every passage, including Chinese text. The output has 384 dimensions and the model input limit is 512 tokens; prefixes and special tokens use part of that budget. Rebuild the collection when the embedding recipe changes.

# Start from the week-17 list of vectors. A vector database adds persistence, identifiers and filters. The fictional Atlas corpus now supplies a versioned technical example. First inspect one document and its metadata, then run the embedding and exact-search path.

import sys
import json
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'sentence-transformers>=3,<6', 'qdrant-client>=1.12,<2', 'transformers==4.56.2'])
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
docs = load_docs()
tasks = json.loads((COURSE / 'data/queries.json').read_text(encoding='utf-8'))
EMBED = 'intfloat/multilingual-e5-small'
encoder = SentenceTransformer(EMBED, device='cpu')
vectors = encoder.encode(['passage: ' + d['title'] + '\n' + d['text'] for d in docs], normalize_embeddings=True)
assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.0001)
print('Vector shape', vectors.shape, 'model max input', encoder.max_seq_length)
path = RUNS / 'week18_qdrant_e5'
client = QdrantClient(path=str(path))
collection = 'atlas_multilingual_e5_small'
if client.collection_exists(collection):
    client.delete_collection(collection)
client.create_collection(collection, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE))
client.upsert(collection, points=[models.PointStruct(id=i, vector=vectors[i].tolist(), payload=d) for i, d in enumerate(docs)])

def semantic_search(question, version='2', k=3):
    q = encoder.encode('query: ' + question, normalize_embeddings=True).tolist()
    filt = None if version == 'all' else models.Filter(must=[models.FieldCondition(key='version', match=models.MatchValue(value=version))])
    return client.query_points(collection_name=collection, query=q, query_filter=filt, limit=k).points
rows = []
lexical = LexicalIndex(docs)
for task in tasks:
    hits = semantic_search(task['question'], task['version'])
    ids = [p.payload['id'] for p in hits]
    row = {'id': task['id'], 'dense': retrieval_metrics(ids, task['relevant'], 3), 'lexical': retrieval_metrics([d['id'] for d in lexical.search(task['question'], task['version'])], task['relevant'], 3), 'dense_ids': ids}
    rows.append(row)
    print(row)
client.close()
client = QdrantClient(path=str(path))
hits = semantic_search('default timeout', '2')
assert hits and all((p.payload['version'] == '2' for p in hits))
print('Persist/reopen/filter check passed')
client.close()
(RUNS / 'week18_retrieval.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


# Update a point and delete another in a separate collection; verify persistence. Save embedding revision and corpus hash. Local mode with nine points is not an ANN or distributed-database performance benchmark.

# ### Reference output
# Expected runtime checks: A nonempty persisted collection; nonempty version-2 hits after reopening; all returned versions equal 2. Real embedding scores vary.
