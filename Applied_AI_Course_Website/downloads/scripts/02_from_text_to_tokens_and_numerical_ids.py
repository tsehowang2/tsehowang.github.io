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


sentence='I go to school by bus'
words=sentence.split();vocab={w:i for i,w in enumerate(words)}
ids=[vocab[w] for w in words]
print(vocab,ids)
inputs,targets=ids[:-1],ids[1:]
print('Inputs:',inputs,'Targets:',targets)
assert inputs[1:]==targets[:-1]
print('Characters:',len(sentence),'UTF-8 bytes:',len(sentence.encode()))


from collections import Counter
corpus = [('b','u','s','_')]*4 + [('b','u','s','y','_')]*2 + [('p','l','u','s','_')]
counts=Counter(pair for word in corpus for pair in zip(word,word[1:]))
print('Pair counts:',counts)
assert counts[('u','s')]==7 and counts[('b','u')]==6
winner=max(counts,key=counts.get);assert winner==('u','s')
def merge(word,pair):
    result=[];i=0
    while i<len(word):
        if i+1<len(word) and (word[i],word[i+1])==pair:result.append(word[i]+word[i+1]);i+=2
        else:result.append(word[i]);i+=1
    return tuple(result)
merged=[merge(w,winner) for w in corpus]
next_counts=Counter(pair for word in merged for pair in zip(word,word[1:]))
assert next_counts[('b','us')]==6
print('First merge:',winner,'Next best:',max(next_counts,key=next_counts.get))


RUN_TOKENIZER=False
if RUN_TOKENIZER:
    import subprocess
    subprocess.check_call([sys.executable,'-m','pip','install','transformers>=4.51,<5'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    examples=['I go to school by bus.','我乘巴士上學。','bus',' bus','bus  bus','257','1,257','１２３']
    for text in examples:
        ids=tokenizer.encode(text,add_special_tokens=False)
        print(repr(text),'UTF-8 bytes',len(text.encode('utf-8')),'model tokens',len(ids),'pieces',tokenizer.convert_ids_to_tokens(ids))
    print('Tokenizer revision',tokenizer.init_kwargs.get('_commit_hash'))
else: print('Tokenizer measurement skipped; no token counts are assumed.')
