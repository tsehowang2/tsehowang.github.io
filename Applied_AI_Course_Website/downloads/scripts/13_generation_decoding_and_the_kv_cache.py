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


import numpy as np
p=np.array([.6,.2,.2]);words=np.array(['bus','train','bicycle'])
rng=np.random.default_rng(7)
print('Greedy:',words[p.argmax()]);print('Twenty samples:',rng.choice(words,20,p=p))
for temperature in [.5,1.,2.]:
    z=np.log(p)/temperature;e=np.exp(z-z.max());print(temperature,e/e.sum())


RUN_MODEL=False
if RUN_MODEL:
    import subprocess
    subprocess.check_call([sys.executable,'-m','pip','install','transformers>=4.51,<5','torch>=2.4,<3'])
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    model_id='Qwen/Qwen3-0.6B'
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id).eval()
    messages=[{'role':'user','content':'Complete with one transport word: I go to school by'}]
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt')
    tokens={'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    with torch.no_grad():out=model.generate(**tokens,max_new_tokens=20,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    print('Revision',model.config._commit_hash,'IDs',tokens['input_ids'].tolist())
    print(tokenizer.decode(out[0],skip_special_tokens=True))
else:print('Real-model generation skipped. Set RUN_MODEL=True to download and run.')
