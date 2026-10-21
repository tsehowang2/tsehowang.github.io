(() => {
  'use strict';
  const $=(q,r)=>r.querySelector(q);
  const $$=(q,r)=>Array.from(r.querySelectorAll(q));
  const factories={};
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fmt=n=>Number(n).toLocaleString('en-US',{maximumFractionDigits:2});
  const text=(r,q,value)=>$(q,r).textContent=value;
  const note=s=>'<p class="viz-note">'+s+'</p>';
  const softmax=xs=>{const m=Math.max(...xs),e=xs.map(x=>Number.isFinite(x)?Math.exp(x-m):0),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);};
  function animateSteps(root,button,last,get,set){
    let timer=null;
    button.addEventListener('click',()=>{
      if(timer){clearInterval(timer);timer=null;button.textContent='Play steps';return;}
      set(0);button.textContent='Pause';
      if(reduced){set(last);button.textContent='Play steps';return;}
      timer=setInterval(()=>{const next=get()+1;set(Math.min(next,last));if(next>=last){clearInterval(timer);timer=null;button.textContent='Play steps';}},1200);
    });
    document.addEventListener('visibilitychange',()=>{if(document.hidden&&timer){clearInterval(timer);timer=null;button.textContent='Play steps';}});
  }

  factories.attention=root=>{
    const labels=['set','request','timeout','now'];
    const Q=[[1,0],[.3,.7],[.6,.4],[0,1]],K=[[1,0],[0,1],[1,1],[.2,.8]],V=[[2,0],[0,4],[1,2],[3,1]];
    let query=2,stage=0;
    root.innerHTML='<div class="viz-controls"><label>Query token <select class="query">'+labels.map((s,i)=>'<option value="'+i+'"'+(i===2?' selected':'')+'>'+s+'</option>').join('')+'</select></label><label><input class="causal" type="checkbox" checked> Causal mask</label></div><div class="viz-display"><div class="viz-step"></div><div class="attention-bars"></div><div class="viz-formula"></div><div class="viz-output" aria-live="polite"></div><div class="table-wrap"><table class="key-table"><caption>Attention weights for every query row</caption><thead><tr><th>Query ↓ / Key →</th>'+labels.map(s=>'<th>'+s+'</th>').join('')+'</tr></thead><tbody></tbody></table></div></div><div class="viz-controls"><button class="previous" type="button">Previous step</button><button class="next" type="button">Next step →</button><button class="play" type="button">Play steps</button></div>'+note('Toy Q, K and V vectors, each with dimension 2. These are designed numerical examples, not learned relationships between these words. Masking changes which positions may contribute; values and downstream layers also matter.');
    $('.attention-bars',root).innerHTML=labels.map((s,i)=>'<div class="weight-row" data-key="'+i+'"><span>'+s+'</span><div class="weight-track"><div class="weight-fill"></div></div><span class="weight-value"></span></div>').join('');
    function weights(row){return softmax(K.map((k,j)=>$('.causal',root).checked&&j>row?-Infinity:(Q[row][0]*k[0]+Q[row][1]*k[1])/Math.sqrt(2)));}
    function render(){
      const causal=$('.causal',root).checked,scores=K.map(k=>(Q[query][0]*k[0]+Q[query][1]*k[1])/Math.sqrt(2)),w=weights(query);
      text(root,'.viz-step',['01 / Score the query against each key','02 / Remove positions the query cannot see','03 / Normalize visible scores with softmax','04 / Mix values using these weights'][stage]);
      $$('.weight-row',root).forEach((row,j)=>{
        const masked=causal&&j>query&&stage>=1;
        row.classList.toggle('masked',masked);
        $('.weight-fill',row).style.width=(masked?0:stage>=2?w[j]*100:scores[j]/1.5*100)+'%';
        $('.weight-value',row).textContent=masked?'masked':stage>=2?w[j].toFixed(3):scores[j].toFixed(3);
      });
      text(root,'.viz-formula',stage<2?'score = (Q · K) / √2':'weights = softmax(masked scores); output = Σ weight × V');
      const out=[0,1].map(d=>w.reduce((sum,p,j)=>sum+p*V[j][d],0));
      text(root,'.viz-output',stage===3?'Mixed output = ('+out.map(x=>x.toFixed(3)).join(', ')+')':stage===2?'Visible weights sum to '+w.reduce((a,b)=>a+b,0).toFixed(3):'Current query: '+labels[query]+' at position '+(query+1));
      $('tbody',root).innerHTML=labels.map((label,i)=>'<tr class="'+(i===query?'active-query':'')+'"><th>'+label+'</th>'+weights(i).map((p,j)=>{
        const masked=causal&&j>i;return '<td style="background:'+(masked?'#e7e9e2':'rgba(45,106,86,'+(.06+.7*p)+')')+';color:'+(p>.65?'white':'#23332f')+'">'+(masked?'—':p.toFixed(2))+'</td>';
      }).join('')+'</tr>').join('');
      $('.previous',root).disabled=stage===0;$('.next',root).disabled=stage===3;
    }
    $('.query',root).addEventListener('change',e=>{query=+e.target.value;render();});
    $('.causal',root).addEventListener('change',render);
    $('.previous',root).addEventListener('click',()=>{stage=Math.max(0,stage-1);render();});
    $('.next',root).addEventListener('click',()=>{stage=Math.min(3,stage+1);render();});
    animateSteps(root,$('.play',root),3,()=>stage,n=>{stage=n;render();});render();
  };

  factories.geometry=root=>{
    root.innerHTML='<div class="viz-controls"><label>Query direction <input class="angle" type="range" min="-30" max="90" value="0"><output class="degrees">0°</output></label><label><input class="normalize" type="checkbox"> Show unit vectors</label></div><div class="viz-display"><div class="geometry-svg"></div><div class="geometry-metrics" aria-live="polite"></div></div>'+note('A = (0.8, 0.6), B = (3, 4). At query = (1, 0), dot product ranks B first, while cosine ranks A first. Unit normalization changes the drawing; the scores compare the original vectors. These are teaching coordinates, not measured model embeddings.');
    function render(){
      const angle=+$('.angle',root).value*Math.PI/180,unit=$('.normalize',root).checked;
      text(root,'.degrees',$('.angle',root).value+'°');
      const q=[Math.cos(angle),Math.sin(angle)],a=unit?[.8,.6]:[.8,.6],b=unit?[.6,.8]:[3,4];
      const width=Math.max(240,Math.round(root.getBoundingClientRect().width-42)),height=230,domain=unit?1.5:5.2;
      const scale=Math.min((width-75)/(domain+1),(height-70)/(domain+1)),ox=40+scale*.4,oy=height*.7;
      const coord=v=>[ox+v[0]*scale,oy-v[1]*scale];
      const vector=(v,label,color,dy)=>{const [x,y]=coord(v);return '<line x1="'+ox+'" y1="'+oy+'" x2="'+x+'" y2="'+y+'" stroke="'+color+'" stroke-width="2.5"/><circle cx="'+x+'" cy="'+y+'" r="4" fill="'+color+'"/><text x="'+(x+7)+'" y="'+(y+dy)+'" style="font-size:12px;fill:'+color+'">'+label+'</text>';};
      const radius=scale;
      $('.geometry-svg',root).innerHTML='<svg viewBox="0 0 '+width+' '+height+'" role="img" aria-label="Two passage vectors differ in direction and length; query direction is adjustable."><line x1="15" y1="'+oy+'" x2="'+(width-15)+'" y2="'+oy+'" stroke="#b4c2b0"/><line x1="'+ox+'" y1="10" x2="'+ox+'" y2="'+(height-13)+'" stroke="#b4c2b0"/><circle cx="'+ox+'" cy="'+oy+'" r="'+radius+'" fill="none" stroke="#c5d2bf" stroke-dasharray="4 4"/>'+vector(b,unit?'B (unit)':'B','#b76738',-7)+vector(a,unit?'A (unit)':'A','#375f82',-7)+vector(q,'query','#2d6a56',17)+'<text x="'+(width-36)+'" y="'+(oy+20)+'" style="font-size:10px">x</text><text x="'+(ox-15)+'" y="15" style="font-size:10px">y</text></svg>';
      const da=q[0]*.8+q[1]*.6,db=q[0]*3+q[1]*4;
      $('.geometry-metrics',root).innerHTML='<span>Dot A <strong>'+da.toFixed(3)+'</strong></span><span>Dot B <strong>'+db.toFixed(3)+'</strong></span><span>Cosine A <strong>'+da.toFixed(3)+'</strong></span><span>Cosine B <strong>'+(db/5).toFixed(3)+'</strong></span>';
    }
    $('.angle',root).addEventListener('input',render);$('.normalize',root).addEventListener('change',render);
    new ResizeObserver(render).observe(root);render();
  };

  factories.generation=root=>{
    const prompt=['Set',' timeout'],generated=['_ms',' to',' 30000','.'];let step=0;
    root.innerHTML='<div class="viz-display"><div class="viz-step"></div><div class="token-stream"></div><div class="viz-formula"></div><div class="generation-bars"></div><div class="viz-output" aria-live="polite"></div></div><div class="viz-controls"><button class="reset" type="button">Reset</button><button class="next" type="button">Generate one token →</button><button class="play" type="button">Play steps</button></div>'+note('Illustrative tokens and probabilities, not the output of a real tokenizer or running model. Generation appends a selected token to the input. During next-token training, the targets are shifted one position to the left.');
    const choices=[['_ms',.7,'_seconds',.2,' error',.1],[' to',.6,' =',.3,' is',.1],[' 30000',.55,' 10000',.35,' default',.1],['.',.8,' ms',.15,' now',.05]];
    function render(){
      text(root,'.viz-step',step<4?'Predict position '+(prompt.length+step+1)+' from the visible prefix':'The bounded example is complete');
      $('.token-stream',root).innerHTML=prompt.map(s=>'<span class="token">'+s+'</span>').join('')+generated.slice(0,step).map((s,i)=>'<span class="token '+(i===step-1?'new':'')+'">'+s+'</span>').join('')+(step<4?'<span class="token pending">next?</span>':'');
      text(root,'.viz-formula','p(next token | all preceding tokens)');
      const c=choices[Math.min(step,3)];
      $('.generation-bars',root).innerHTML=[0,2,4].map(i=>'<div class="weight-row"><span>'+c[i]+'</span><div class="weight-track"><div class="weight-fill" style="width:'+(100*c[i+1])+'%"></div></div><span class="weight-value">'+Math.round(100*c[i+1])+'%</span></div>').join('');
      text(root,'.viz-output',step===4?'Generated continuation: '+generated.join(''):'Greedy choice in this example: '+c[0]);
      $('.next',root).disabled=step===4;
    }
    $('.next',root).addEventListener('click',()=>{step=Math.min(4,step+1);render();});
    $('.reset',root).addEventListener('click',()=>{step=0;render();});
    animateSteps(root,$('.play',root),4,()=>step,n=>{step=n;render();});render();
  };

  factories.lora=root=>{
    let stage=0;
    root.innerHTML='<div class="viz-controls"><label>Adapter rank r <input class="rank" type="range" min="1" max="64" value="8"><output class="rank-value">8</output></label><button class="step" type="button">Follow a training step →</button></div><div class="viz-display"><div class="viz-step"></div><div class="lora-path"><div class="matrix frozen"><strong>W</strong>1024 × 1024<small>Frozen base weights</small></div><div class="lora-chain"><div class="matrix trainable a"><strong>A</strong><span></span><small>Trainable low rank factor</small></div><div class="matrix trainable b"><strong>B</strong><span></span><small>Trainable low rank factor</small></div></div></div><div class="viz-formula">y = Wx + (α/r) BAx</div><div class="viz-output" aria-live="polite"></div><div class="lora-count"></div></div>'+note('One illustrative dense layer. Rank changes the adapter parameter count, not a guarantee of better quality. LoRA still stores the frozen base and needs activations. This illustration follows gradient routing; it does not train a model.');
    function render(){
      const r=+$('.rank',root).value;
      text(root,'.rank-value',r);text(root,'.a span',r+' × 1024');text(root,'.b span','1024 × '+r);
      text(root,'.viz-step',['01 / Forward: combine base and adapter paths','02 / Backward: gradients reach the adapter factors','03 / Update: A and B change; W stays frozen'][stage]);
      $$('.trainable',root).forEach(e=>e.classList.toggle('highlight',stage>0));
      text(root,'.viz-output',fmt(2048*r)+' trainable parameters / '+fmt(1024*1024)+' frozen base parameters');
      text(root,'.lora-count',(100*2048*r/(1024*1024)).toFixed(3)+'% of the dense layer parameter count');
      $('.lora-count',root).style.fontSize='11px';
      $('.step',root).textContent=stage===2?'Restart the training step':'Follow a training step →';
    }
    $('.rank',root).addEventListener('input',render);$('.step',root).addEventListener('click',()=>{stage=(stage+1)%3;render();});render();
  };

  factories.rag=root=>{
    let step=0;
    const names=['Question','Filter versions','Retrieve evidence','Pack context','Form the answer','Validate citations'];
    root.innerHTML='<div class="viz-controls"><label>Requested version <select class="version"><option value="2">Version 2</option><option value="1">Version 1</option><option value="all">Compare versions 1 and 2</option></select></label><label><input class="missing" type="checkbox"> Remove necessary evidence</label></div><div class="viz-display"><div class="rag-question"></div><div class="flow-grid">'+names.map((n,i)=>'<div class="flow-node"><span>0'+(i+1)+'</span><strong>'+n+'</strong></div>').join('')+'</div><div class="evidence-list"><div class="evidence v1"><small>[timeout-v1] / version 1</small>Default: 10000 milliseconds</div><div class="evidence v2"><small>[timeout-v2] / version 2</small>Default: 30000 milliseconds</div></div><div class="viz-step"></div><div class="viz-output" aria-live="polite"></div></div><div class="viz-controls"><button class="previous" type="button">Previous stage</button><button class="next" type="button">Next stage →</button><button class="play" type="button">Play steps</button></div>'+note('A deterministic walkthrough using the fictional Atlas documentation. No model is running. Existing citation IDs establish structure, not semantic support; the answer must still be checked against the cited facts.');
    function render(){
      const v=$('.version',root).value,missing=$('.missing',root).checked;
      text(root,'.rag-question',v==='all'?'Question: How did the timeout default change?':'Question: What is the timeout default in version '+v+'?');
      $('.rag-question',root).style.fontSize='12px';
      $$('.flow-node',root).forEach((e,i)=>e.classList.toggle('active',i===step));
      const exclude1=step>=1&&v==='2'||step>=2&&missing&&(v==='1'||v==='all');
      const exclude2=step>=1&&v==='1'||step>=2&&missing&&v==='2';
      $('.v1',root).classList.toggle('excluded',exclude1);$('.v2',root).classList.toggle('excluded',exclude2);
      const message=[
        'Keep the requested version as structured task state.',
        v==='all'?'Both versions are eligible for this comparison.':'Only version '+v+' sources are eligible.',
        missing?'A required supporting passage is unavailable.':'Required supporting passages were found.',
        'Label each passage with its source ID and version; reserve output tokens.',
        missing?'Insufficient evidence to answer the complete question.':v==='all'?'The default changed from 10000 ms [timeout-v1] to 30000 ms [timeout-v2].':v==='1'?'The default is 10000 ms [timeout-v1].':'The default is 30000 ms [timeout-v2].',
        missing?'Acknowledge the missing evidence; do not invent a comparison.':'Check allowed source IDs, then inspect whether each claim follows from them.'
      ];
      text(root,'.viz-step','0'+(step+1)+' / '+names[step]);
      text(root,'.viz-output',message[step]);
      $('.previous',root).disabled=step===0;$('.next',root).disabled=step===5;
    }
    $('.version',root).addEventListener('change',()=>{step=0;render();});$('.missing',root).addEventListener('change',render);
    $('.previous',root).addEventListener('click',()=>{step=Math.max(0,step-1);render();});$('.next',root).addEventListener('click',()=>{step=Math.min(5,step+1);render();});
    animateSteps(root,$('.play',root),5,()=>step,n=>{step=n;render();});render();
  };

  factories.retrieval=root=>{
    root.innerHTML='<div class="viz-controls"><label>RRF constant c <input class="c" type="range" min="0" max="60" value="60"><output class="c-value">60</output></label><label>Returned candidates k <input class="k" type="range" min="1" max="5" value="2"><output class="k-value">2</output></label><label><input class="missing" type="checkbox"> Remove passage A</label></div><div class="viz-display"><div class="two-rankings"><div><h3>Lexical candidates</h3><ol class="lexical"></ol></div><div><h3>Dense candidates</h3><ol class="dense"></ol></div></div><div class="viz-formula">RRF(document) = Σ 1 / (c + rank)</div><div class="fusion-bars"></div><div class="metric-readout"></div><div class="viz-output" aria-live="polite"></div></div>'+note('Designed rank lists, not measured model results. A and B are both required evidence. Items absent from a candidate list contribute zero from that list. Reranking cannot restore a passage missing from every candidate list.');
    function render(){
      const c=+$('.c',root).value,k=+$('.k',root).value,missing=$('.missing',root).checked;
      const lexical=['A','B'].filter(x=>!(missing&&x==='A'));
      const dense=['C','B','D','E','F','G','H','I','J','A'].filter(x=>!(missing&&x==='A'));
      text(root,'.c-value',c);text(root,'.k-value',k);
      $('.lexical',root).innerHTML=lexical.map(x=>'<li>Passage '+x+(x==='A'||x==='B'?' · relevant':'')+'</li>').join('');
      $('.dense',root).innerHTML=dense.map(x=>'<li>Passage '+x+(x==='A'||x==='B'?' · relevant':'')+'</li>').join('');
      const score={};[lexical,dense].forEach(list=>list.forEach((x,i)=>score[x]=(score[x]||0)+1/(c+i+1)));
      const ranking=Object.entries(score).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])),top=ranking.slice(0,k),max=ranking[0][1];
      $('.fusion-bars',root).innerHTML=top.map(([id,value])=>'<div class="weight-row"><span>Passage '+id+'</span><div class="weight-track"><div class="weight-fill" style="width:'+value/max*100+'%;background:'+(id==='A'||id==='B'?'#2d6a56':'#8b9c87')+'"></div></div><span class="weight-value">'+value.toFixed(4)+'</span></div>').join('');
      const recalled=top.filter(([id])=>id==='A'||id==='B').length;
      text(root,'.metric-readout','Evidence Recall@'+k+' = '+recalled+'/2');
      text(root,'.viz-output',recalled===2?'Both required passages are available to the answerer.':'The answerer lacks '+(2-recalled)+' required passage'+(recalled===0?'s':'')+'.');
    }
    $$('.viz-controls input',root).forEach(e=>e.addEventListener('input',render));render();
  };

  factories.memory=root=>{
    const colors=['#2d6a56','#63879a','#b2904c','#9b725b','#b2be9d'];
    root.innerHTML='<div class="viz-controls"><label>Mode <select class="mode"><option value="inference">Inference</option><option value="training">Full training</option></select></label><label>Parameters <input class="params" type="range" min="100" max="3000" step="100" value="500"><output class="p-value">500M</output></label><label>Sequence tokens <input class="tokens" type="range" min="128" max="4096" step="128" value="2048"><output class="t-value">2048</output></label><label>Batch <select class="batch"><option>1</option><option>2</option><option>4</option></select></label><label>Illustrative VRAM <select class="capacity"><option value="4">4 GiB</option><option value="16">16 GiB</option></select></label></div><div class="viz-display"><div class="viz-step"></div><div class="memory-track"></div><div class="capacity-label"></div><div class="viz-legend"></div><div class="metric-readout"></div><div class="viz-output" aria-live="polite"></div><div class="viz-formula"></div></div>'+note('A partial resource worksheet, not a measured fit prediction. Inference includes 2-byte weights plus a KV cache with 24 layers, 8 KV heads and head dimension 64. Example full training state uses 16 bytes per parameter. Activations, temporary buffers, allocator reserve and other runtime costs are omitted. Parameter count and cache geometry are independent worksheet inputs.');
    function render(){
      const p=+$('.params',root).value*1e6,t=+$('.tokens',root).value,b=+$('.batch',root).value,training=$('.mode',root).value==='training',capacity=+$('.capacity',root).value*2**30;
      text(root,'.p-value',fmt(p/1e6)+'M');text(root,'.t-value',t);
      const parts=training?[['Weights',2*p],['Gradients',2*p],['Master weights',4*p],['Optimizer moments',8*p]]:[['Weights',2*p],['KV cache',2*24*b*t*8*64*2]];
      const total=parts.reduce((sum,x)=>sum+x[1],0),domain=Math.max(total,capacity)*1.08;
      $('.memory-track',root).innerHTML=parts.map(([name,value],i)=>'<div class="memory-part" style="width:'+100*value/domain+'%;background:'+colors[i]+'" aria-label="'+name+' '+(value/2**30).toFixed(2)+' GiB"></div>').join('')+'<span class="capacity-line" style="left:'+100*capacity/domain+'%"></span>';
      $('.viz-legend',root).innerHTML=parts.map(([name,value],i)=>'<span><i style="background:'+colors[i]+'"></i>'+name+' '+(value/2**30).toFixed(2)+' GiB</span>').join('');
      text(root,'.capacity-label','Orange marker: chosen '+$('.capacity',root).value+' GiB budget');
      text(root,'.viz-step',training?'Training keeps more than model weights':'Inference reuses earlier keys and values');
      text(root,'.metric-readout',(total/2**30).toFixed(2)+' GiB of counted state');
      text(root,'.viz-output',total>capacity?'Already above the selected budget before omitted runtime costs.':'Below the budget in this partial estimate; omitted runtime costs still need headroom.');
      text(root,'.viz-formula',training?'State = parameters × (2 + 2 + 4 + 8) bytes':'KV bytes = 2 × layers × batch × tokens × KV heads × head dimension × 2');
    }
    $$('input,select',root).forEach(e=>e.addEventListener('input',render));render();
  };

  factories.agent=root=>{
    let phase=0,round=0,finished=false,status='ready',proposal={},observation='',trace=[],timer=null;
    const names=['Propose an action','Validate schema and scope','Execute the allowed tool','Record the observation','Finish or stop'];
    root.innerHTML='<div class="viz-controls"><label>Scenario <select class="scenario"><option value="success">Lookup and calculate</option><option value="unknown">Unknown tool</option><option value="version">Wrong version</option><option value="loop">Planner never stops</option><option value="config">Invalid configuration</option></select></label><button class="reset" type="button">Reset trace</button><button class="next" type="button">Next step →</button><button class="play" type="button">Play trace</button></div><div class="viz-display"><div class="flow-grid">'+names.map((name,i)=>'<div class="flow-node"><span>0'+(i+1)+'</span><strong>'+name+'</strong></div>').join('')+'</div><div class="viz-step"></div><div class="agent-proposal"></div><ol class="agent-trace"></ol><div class="viz-output agent-status" aria-live="polite"></div></div>'+note('A deterministic controller simulation, not a running model. The request allows version 2 and at most three proposals. A valid validator call can return an invalid configuration. Step limits do not interrupt a blocking call, and accepted final text still needs task-level checking.');
    function scenario(){return $('.scenario',root).value;}
    function propose(){
      if(scenario()==='unknown')return {type:'tool',name:'export_secrets',arguments:{}};
      if(scenario()==='version')return {type:'tool',name:'lookup',arguments:{query:'retry policy',version:'1'}};
      if(scenario()==='loop')return {type:'tool',name:'calculate',arguments:{expression:'1+1'}};
      if(scenario()==='config')return round===0?{type:'tool',name:'validate_config',arguments:{config:{timeout_ms:0,retries:3}}}:{type:'final',answer:'Configuration is invalid: timeout_ms is outside the allowed range.',citations:[]};
      if(round===0)return {type:'tool',name:'lookup',arguments:{query:'retry backoff',version:'2'}};
      if(round===1)return {type:'tool',name:'calculate',arguments:{expression:'100+200+400'}};
      return {type:'final',answer:'Cumulative waiting time is 700 ms before attempt four.',citations:['retry-v2']};
    }
    function render(){
      $$('.flow-node',root).forEach((e,i)=>e.classList.toggle('active',i===phase));
      text(root,'.viz-step','Proposal '+Math.min(round+1,3)+' of 3 · '+names[phase]);
      text(root,'.agent-proposal',phase===0?'The planner is ready to propose the next action.':JSON.stringify(proposal,null,2));
      $('.agent-trace',root).replaceChildren();
      trace.forEach(line=>{const li=document.createElement('li');li.textContent=line;$('.agent-trace',root).append(li);});
      text(root,'.agent-status',status);
      $('.next',root).disabled=finished;
      if(finished&&timer){clearInterval(timer);timer=null;$('.play',root).textContent='Play trace';}
    }
    function advance(){
      if(finished)return;
      if(phase===0){proposal=propose();status='proposed; nothing executed';phase=1;}
      else if(phase===1){
        if(proposal.name==='export_secrets'){status='invalid_action: unknown tool; execution prevented';trace.push('Rejected unknown tool before execution.');finished=true;phase=4;}
        else if(proposal.name==='lookup'&&proposal.arguments.version!=='2'){status='invalid_action: request scope allows version 2 only';trace.push('Rejected wrong-version lookup before execution.');finished=true;phase=4;}
        else if(proposal.type==='final'){status='completed: '+proposal.answer;trace.push('Final contract accepted; independent task checking is still required.');finished=true;phase=4;}
        else{status='validated: allowed tool and arguments';phase=2;}
      }else if(phase===2){
        observation=proposal.name==='lookup'?'[retry-v2] Wait 100, 200, then 400 ms.':proposal.name==='validate_config'?'valid=false; timeout_ms must be 1..120000':scenario()==='loop'?'value=2':'value=700';
        status='tool returned '+observation;phase=3;
      }else if(phase===3){
        trace.push('Action '+(round+1)+': '+proposal.name+' → '+observation);round++;
        if(round>=3){status='budget_exhausted: stopped after three allowed proposals';finished=true;phase=4;}
        else{status='observation recorded; select the next action';phase=0;}
      }
      render();
    }
    function reset(){if(timer)clearInterval(timer);timer=null;phase=0;round=0;finished=false;status='ready';proposal={};observation='';trace=[];$('.play',root).textContent='Play trace';render();}
    $('.next',root).addEventListener('click',advance);$('.reset',root).addEventListener('click',reset);$('.scenario',root).addEventListener('change',reset);
    $('.play',root).addEventListener('click',()=>{
      if(timer){clearInterval(timer);timer=null;$('.play',root).textContent='Play trace';return;}
      if(finished)reset();
      if(reduced){for(let i=0;i<30&&!finished;i++)advance();return;}
      $('.play',root).textContent='Pause';advance();if(!finished)timer=setInterval(advance,1050);
    });
    document.addEventListener('visibilitychange',()=>{if(document.hidden&&timer){clearInterval(timer);timer=null;$('.play',root).textContent='Play trace';}});
    render();
  };

  factories.diffusion=root=>{
    const original=Array.from({length:64},(_,i)=>{const r=Math.floor(i/8),c=i%8;return (r===1&&c>=1&&c<=6||c===3&&r>=1&&r<=6)?1:-1;});
    let seed=23;
    function uniform(){seed=(1664525*seed+1013904223)>>>0;return (seed+.5)/4294967296;}
    const noise=original.map(()=>Math.sqrt(-2*Math.log(uniform()))*Math.cos(2*Math.PI*uniform()));
    let timer=null;
    root.innerHTML='<div class="viz-controls"><label>Noise level <input class="noise" type="range" min="0" max="100" step="1" value="0"><output class="noise-value">0%</output></label><button class="play" type="button">Play forward noising</button><button class="reset" type="button">Reset</button></div><div class="viz-display"><div class="noise-figures"><figure><div class="noise-grid clean"></div><figcaption>Original signal x₀</figcaption></figure><figure><div class="noise-grid mixed"></div><figcaption>Noised sample xₜ</figcaption></figure></div><div class="viz-formula">xₜ = √ᾱ · x₀ + √(1 − ᾱ) · ε</div><div class="viz-output" aria-live="polite"></div></div>'+note('An 8 × 8 designed signal and one fixed seeded Gaussian noise sample. The slider changes ᾱ from 1 to 0. This illustrates forward corruption only; moving the slider backward is not a learned reverse diffusion model. Colors clip display values to [−2, 2].');
    function color(value){const t=Math.max(0,Math.min(1,(value+2)/4));const low=[239,240,223],high=[23,60,54];return 'rgb('+low.map((x,i)=>Math.round(x+(high[i]-x)*t)).join(',')+')';}
    $('.clean',root).innerHTML=original.map(v=>'<span class="noise-cell" style="background:'+color(v)+'"></span>').join('');
    $('.mixed',root).innerHTML=original.map(()=>'<span class="noise-cell"></span>').join('');
    function render(){
      const level=+$('.noise',root).value/100,alpha=1-level;
      text(root,'.noise-value',Math.round(100*level)+'%');
      $$('.mixed .noise-cell',root).forEach((e,i)=>e.style.background=color(Math.sqrt(alpha)*original[i]+Math.sqrt(1-alpha)*noise[i]));
      text(root,'.viz-output','Signal coefficient '+Math.sqrt(alpha).toFixed(2)+' · noise coefficient '+Math.sqrt(1-alpha).toFixed(2));
    }
    function stop(){if(timer)clearInterval(timer);timer=null;$('.play',root).textContent='Play forward noising';}
    $('.noise',root).addEventListener('input',()=>{stop();render();});
    $('.reset',root).addEventListener('click',()=>{stop();$('.noise',root).value=0;render();});
    $('.play',root).addEventListener('click',()=>{
      if(timer){stop();return;}
      $('.noise',root).value=0;render();
      if(reduced){$('.noise',root).value=100;render();return;}
      $('.play',root).textContent='Pause';
      timer=setInterval(()=>{const n=Math.min(100,+$('.noise',root).value+10);$('.noise',root).value=n;render();if(n===100)stop();},350);
    });
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});render();
  };

  document.querySelectorAll('[data-widget]').forEach(root=>{
    const factory=factories[root.dataset.widget];
    if(factory)factory(root);
  });
  window.AppliedAIVisuals={softmax,widgetTypes:Object.keys(factories)};
})();
