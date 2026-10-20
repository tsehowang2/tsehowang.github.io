(() => {
  'use strict';
  const $=(s,r)=>r.querySelector(s);
  const bar=(label,p)=>`<div class="beginner-bar"><span>${label}</span><div><i style="width:${p*100}%"></i></div><strong>${(p*100).toFixed(1)}%</strong></div>`;
  document.querySelectorAll('[data-widget="prediction"]').forEach(root=>{
    root.innerHTML='<p class="sentence-example">I go to school by <strong>___</strong></p><label>Bus examples <input class="bus-count" type="range" min="0" max="10" value="3"></label><label>Train examples <input class="train-count" type="range" min="0" max="10" value="1"></label><p>One bicycle example stays fixed.</p><div class="prediction-bars" aria-live="polite"></div><p class="visual-caption">Observed proportions in an invented dataset. These are not real commuting statistics or trained-model outputs.</p>';
    function render(){const b=+$('.bus-count',root).value,t=+$('.train-count',root).value,n=b+t+1;$('.prediction-bars',root).innerHTML=`<p>${b} bus + ${t} train + 1 bicycle = ${n} matching sentences</p>`+bar('bus',b/n)+bar('train',t/n)+bar('bicycle',1/n);}
    root.querySelectorAll('input').forEach(x=>x.addEventListener('input',render));render();
  });
  document.querySelectorAll('[data-widget="tokens"]').forEach(root=>{
    const words=['I','go','to','school','by','bus'];
    root.innerHTML='<p class="visual-caption">Teaching tokenizer: one word per token. A real model may use subwords and separate punctuation.</p><div class="token-buttons">'+words.map((w,i)=>`<button type="button" data-token="${i}">${w}<small>ID ${i}</small></button>`).join('')+'</div><div class="token-detail" aria-live="polite"></div>';
    function render(i){root.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.token===i)));$('.token-detail',root).innerHTML=`<p><strong>${words[i]}</strong> → ID ${i} → look up embedding table row ${i}.</p><p>${i<5?`The prefix “${words.slice(0,i+1).join(' ')}” trains the next-token target <strong>${words[i+1]}</strong>.`:'This sentence has ended. Another token or end marker would be needed for a following target.'}</p><p>The ID is an index. ID 5 does not mean bus has five times the meaning of go.</p>`;}
    root.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>render(+b.dataset.token)));render(0);
  });
  document.querySelectorAll('[data-widget="probabilities"]').forEach(root=>{
    root.innerHTML='<p class="sentence-example">I go to school by ___</p><label>Bus score <input class="bus-score" type="range" min="-3" max="5" step=".1" value="2"></label><label>Temperature <input class="temperature" type="range" min=".2" max="3" step=".1" value="1"></label><div class="probability-bars" aria-live="polite"></div><p class="visual-caption">Train score = 1; bicycle score = 0. Divide all scores by temperature, subtract their maximum, exponentiate, then normalize.</p>';
    function render(){const z=[+$('.bus-score',root).value,1,0],t=+$('.temperature',root).value,m=Math.max(...z),e=z.map(v=>Math.exp((v-m)/t)),sum=e.reduce((a,b)=>a+b,0),p=e.map(v=>v/sum);$('.probability-bars',root).innerHTML=`<p>Scores [${z.map(v=>v.toFixed(1)).join(', ')}], temperature ${t.toFixed(1)}</p>`+p.map((v,i)=>bar(['bus','train','bicycle'][i],v)).join('')+`<p>If train is the target: −log p(train) = <strong>${(-Math.log(p[1])).toFixed(3)} nats</strong>.</p>`;}
    root.querySelectorAll('input').forEach(x=>x.addEventListener('input',render));render();
  });
  document.querySelectorAll('[data-widget="mixture"]').forEach(root=>{
    root.innerHTML='<p>Three information cards carry values [2, 0], [0, 4] and [2, 2]. We supply the weights before learning where weights come from.</p><label>First-card weight <input class="mix-weight" type="range" min="0" max="100" value="50"></label><div class="mix-result" aria-live="polite"></div><p class="visual-caption">The remaining weight is split 60% to card two and 40% to card three. These invented coordinates are payload values, not probabilities.</p>';
    function render(){const a=+$('.mix-weight',root).value/100,b=(1-a)*.6,c=(1-a)*.4;$('.mix-result',root).innerHTML=bar('card 1',a)+bar('card 2',b)+bar('card 3',c)+`<p>First coordinate: ${a.toFixed(2)} × 2 + ${b.toFixed(2)} × 0 + ${c.toFixed(2)} × 2 = <strong>${(2*a+2*c).toFixed(2)}</strong></p><p>Second coordinate: ${a.toFixed(2)} × 0 + ${b.toFixed(2)} × 4 + ${c.toFixed(2)} × 2 = <strong>${(4*b+2*c).toFixed(2)}</strong></p>`;}
    $('.mix-weight',root).addEventListener('input',render);render();
  });
  document.querySelectorAll('[data-widget="block"]').forEach(root=>{
    const steps=[['Input residual stream','Token and position information arrives as a T × D matrix.'],['Normalize, then attend','For each token, attention mixes information from allowed token positions. The attention output has the same T × D shape.'],['First residual addition','Add the attention result to the incoming stream: y = x + Attention(LN(x)).'],['Normalize, then process features','A positionwise MLP expands and transforms each token’s feature vector independently.'],['Second residual addition','z = y + MLP(LN(y)). This completes one pre-normalized block. Stack blocks, then normalize and project to vocabulary logits.']];
    root.innerHTML='<div class="block-diagram">'+steps.map((s,i)=>`<button type="button" data-block-step="${i}"><span>${i+1}</span>${s[0]}</button>`).join('')+'</div><div class="block-explanation" aria-live="polite"></div>';
    function show(i){root.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.blockStep===i)));$('.block-explanation',root).innerHTML=`<h3>${steps[i][0]}</h3><p>${steps[i][1]}</p><p>Trace the shape T × D before moving to the next stage.</p>`;}
    root.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>show(+b.dataset.blockStep)));show(0);
  });
})();
