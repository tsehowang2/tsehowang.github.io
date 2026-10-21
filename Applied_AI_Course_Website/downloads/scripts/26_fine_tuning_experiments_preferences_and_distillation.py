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


beta=.1;relative_margin=.4-.1;scaled_margin=beta*relative_margin;dpo_loss=math.log1p(math.exp(-scaled_margin))
assert abs(dpo_loss-.6782596763414483)<1e-12
print('Illustrative DPO loss (not DPO training)',dpo_loss)


import subprocess
subprocess.check_call([sys.executable,'-m','pip','install','-q','torch>=2.4,<3','transformers>=4.51,<5','peft>=0.13,<1'])
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model,PeftModel
if not (RUNS/'week24_train.json').exists(): raise RuntimeError('Run tutorial 24 first.')
torch.manual_seed(7);device='cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID='Qwen/Qwen3-0.6B';tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
# Float32 keeps this beginner training path simple. Use a small model and short input.
base=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32).to(device)
base.config.use_cache=False;base_revision=base.config._commit_hash
model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=0,
                                    target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
model.print_trainable_parameters()


def prepare(record,max_length=384):
    messages=record['messages']
    prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=False)
    if full[:len(prefix)]!=prefix: raise ValueError('Template prefix mismatch: use a proper assistant-token mask.')
    ids=full[:max_length];labels=ids.copy()
    for i in range(min(len(prefix),len(labels))): labels[i]=-100
    if all(v==-100 for v in labels): raise ValueError('All answer tokens truncated: shorten prompt or increase limit.')
    return {'input_ids':torch.tensor([ids],device=device),
            'attention_mask':torch.ones(1,len(ids),dtype=torch.long,device=device),
            'labels':torch.tensor([labels],device=device)}
train=json.loads((RUNS/'week24_train.json').read_text());dev=json.loads((RUNS/'week24_dev.json').read_text())
train_batches=[prepare(r) for r in train];dev_batches=[prepare(r) for r in dev]
def score(batches,disable=False):
    model.eval();total,n=0.,0
    context=model.disable_adapter() if disable else __import__('contextlib').nullcontext()
    with context,torch.no_grad():
        for b in batches:
            count=int((b['labels'][:,1:]!=-100).sum())
            total+=model(**b).loss.item()*count;n+=count
    return total/n
before=score(dev_batches,disable=True)
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
steps=20 if device=='cuda' else 2
if device=='cuda': torch.cuda.reset_peak_memory_stats()
start=time.perf_counter();losses=[];model.train()
for step in range(steps):
    b=train_batches[step%len(train_batches)];optimizer.zero_grad();loss=model(**b).loss
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
    loss.backward()
    if step==0:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
    optimizer.step();losses.append(loss.item());print(step,loss.item())
after=score(dev_batches)
adapter=RUNS/'week26_adapter';model.save_pretrained(adapter);tokenizer.save_pretrained(adapter)
report={'base_model':MODEL_ID,'base_revision':base_revision,'steps':steps,'device':device,
        'base_dev_nll':before,'adapter_dev_nll':after,'losses':losses,'seconds':time.perf_counter()-start,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}
(RUNS/'week26_training.json').write_text(json.dumps(report,indent=2));print(report)


# Release the first model before reloading to avoid two full GPU copies.
del optimizer,train_batches,dev_batches,model,base,loss
import gc;gc.collect()
if device=='cuda': torch.cuda.empty_cache()
reloaded_base=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=base_revision,torch_dtype=torch.float32).to(device)
reloaded=PeftModel.from_pretrained(reloaded_base,adapter).eval()
sample=prepare(dev[0])
with torch.no_grad(): reload_loss=reloaded(**sample).loss.item()
assert math.isfinite(reload_loss)
print('Reload finite-loss smoke check',reload_loss)
