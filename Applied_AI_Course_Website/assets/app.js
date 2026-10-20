(() => {
  'use strict';
  const base=window.COURSE_BASE||'';
  const $=(q,r=document)=>r.querySelector(q);
  const $$=(q,r=document)=>Array.from(r.querySelectorAll(q));
  const store={
    get(key,fallback){try{const v=localStorage.getItem('applied-ai:'+key);return v===null?fallback:JSON.parse(v);}catch{return fallback;}},
    set(key,value){try{localStorage.setItem('applied-ai:'+key,JSON.stringify(value));return true;}catch{return false;}}
  };
  let completed=store.get('completed',[]);
  if(!Array.isArray(completed))completed=[];
  completed=completed.filter(n=>Number.isInteger(n)&&n>=1&&n<=32);
  function updateProgress(){
    $$('[data-week-tick]').forEach(e=>{e.textContent=completed.includes(+e.dataset.weekTick)?'✓':'';e.setAttribute('aria-label',e.textContent?'Completed':'Not completed');});
    $$('[data-progress-count]').forEach(e=>e.textContent=completed.length+' / 32');
    $$('.side-progress progress').forEach(e=>e.value=completed.length);
    $$('.mark-complete').forEach(b=>{const yes=completed.includes(+b.dataset.week);b.setAttribute('aria-pressed',String(yes));b.textContent=yes?'Completed ✓':'Mark as complete';});
  }
  let toastTimer;
  function toast(text){const e=$('.toast');e.textContent=text;e.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>e.classList.remove('show'),2200);}
  $$('.mark-complete').forEach(b=>b.addEventListener('click',()=>{const n=+b.dataset.week;completed=completed.includes(n)?completed.filter(x=>x!==n):[...completed,n];const saved=store.set('completed',completed);updateProgress();toast(saved?'Progress saved in this browser.':'Progress updated for this page; browser storage is unavailable.');}));
  updateProgress();
  $$('[data-answer-key]').forEach(e=>{e.value=store.get('answer:'+e.dataset.answerKey,'');e.addEventListener('input',()=>store.set('answer:'+e.dataset.answerKey,e.value));});
  const menu=$('.menu-toggle');
  menu.addEventListener('click',()=>{const open=document.body.classList.toggle('menu-open');menu.setAttribute('aria-expanded',String(open));});
  document.addEventListener('click',e=>{if(document.body.classList.contains('menu-open')&&!e.target.closest('.sidebar,.menu-toggle')){document.body.classList.remove('menu-open');menu.setAttribute('aria-expanded','false');}});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.body.classList.remove('menu-open');menu.setAttribute('aria-expanded','false');}});
  $$('.copy-code').forEach(button=>button.addEventListener('click',async()=>{
    const text=$('pre code',button.closest('.code-block')).textContent;
    try{
      if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(text);
      else{const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.left='-9999px';document.body.append(area);area.select();const ok=document.execCommand('copy');area.remove();if(!ok)throw Error('Copy failed');}
      button.textContent='Copied';setTimeout(()=>button.textContent='Copy code',1600);
    }catch{toast('Select the code and copy it using your keyboard.');}
  }));
  const dialog=$('.search-dialog'),input=$('#course-search'),results=$('.search-results'),status=$('.search-status');
  function openSearch(){if(!dialog.open)dialog.showModal();input.focus();}
  $('.search-open').addEventListener('click',openSearch);
  $('.search-close').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
  document.addEventListener('keydown',e=>{if(e.key==='/'&&!e.ctrlKey&&!e.metaKey&&!/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)){e.preventDefault();openSearch();}});
  input.addEventListener('input',()=>{
    const query=input.value.trim().toLowerCase();results.replaceChildren();
    if(query.length<2){status.textContent='Type at least two characters to search the course.';return;}
    const terms=query.split(/\s+/);
    const matches=(window.COURSE_SEARCH||[]).map(item=>{
      const title=item.title.toLowerCase(),body=item.text.toLowerCase();
      if(!terms.every(t=>title.includes(t)||body.includes(t)))return null;
      return {...item,score:terms.reduce((s,t)=>s+(title.includes(t)?12:0)+(body.includes(t)?1:0),0)};
    }).filter(Boolean).sort((a,b)=>b.score-a.score).slice(0,20);
    status.textContent=matches.length?matches.length+' matching pages'+(matches.length===20?' (first 20)':''):'No matches. Try a shorter concept or a synonym.';
    matches.forEach(item=>{
      const a=document.createElement('a');a.className='search-result';a.href=base+item.path;
      const type=document.createElement('small');type.textContent=item.type;
      const title=document.createElement('strong');title.textContent=item.title;
      const snippet=document.createElement('p');const clean=item.text.replace(/\s+/g,' ').trim();const at=clean.toLowerCase().indexOf(terms[0]);const start=Math.max(0,at-65);
      snippet.textContent=(start?'…':'')+clean.slice(start,start+190)+'…';a.append(type,title,snippet);results.append(a);
    });
  });
  $$('.colab-link').forEach(a=>{
    const config=window.COURSE_SITE||{};
    if(config.githubOwner&&config.githubRepo){
      const path=(config.repositoryPath||'').replace(/^\/|\/$/g,'');
      a.href='https://colab.research.google.com/github/'+encodeURIComponent(config.githubOwner)+'/'+encodeURIComponent(config.githubRepo)+'/blob/'+encodeURIComponent(config.githubBranch||'main')+'/'+(path?path+'/':'')+a.dataset.notebook;
    }else if(location.hostname.endsWith('.github.io')){
      const owner=location.hostname.split('.')[0],segments=location.pathname.split('/').filter(Boolean);
      const isUserSite=segments[0]==='lectures'||segments[0]==='tutorials'||segments[0]==='tests'||segments.length<=1;
      const repo=isUserSite?owner+'.github.io':segments[0];
      a.href='https://colab.research.google.com/github/'+encodeURIComponent(owner)+'/'+encodeURIComponent(repo)+'/blob/'+encodeURIComponent(config.githubBranch||'main')+'/'+a.dataset.notebook;
    }
  });
})();
