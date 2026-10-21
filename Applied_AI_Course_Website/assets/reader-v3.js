(() => {
  'use strict';
  const modes=[...document.querySelectorAll('[data-mode]')];
  if(!modes.length)return;
  const reader=document.querySelector('#full-notes'),deck=document.querySelector('.lecture-deck'),toc=document.querySelector('.reader-toc'),visual=document.querySelector('.visual-lesson');
  const select=mode=>{
    const slides=mode==='slides';
    reader.hidden=slides;deck.hidden=!slides;if(toc)toc.hidden=slides;if(visual)visual.hidden=slides;
    modes.forEach(b=>b.setAttribute('aria-selected',String(b.dataset.mode===mode)));
    modes.forEach(b=>b.tabIndex=b.dataset.mode===mode?0:-1);
    if(!slides&&deck.classList.contains('presenting'))deck.querySelector('.deck-present').click();
  };
  modes.forEach(b=>b.addEventListener('click',()=>select(b.dataset.mode)));
  modes.forEach((b,i)=>b.addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?modes.length-1:(i+(event.key==='ArrowRight'?1:-1)+modes.length)%modes.length;
    select(modes[next].dataset.mode);modes[next].focus();
  }));
  document.querySelectorAll('.slide-section-link').forEach(a=>a.addEventListener('click',()=>{select('read');if(a.dataset.openReference)document.querySelector('#reference-details').open=true;}));
  document.querySelectorAll('.reader-toc a').forEach(a=>a.addEventListener('click',()=>select('read')));
  const readHash=()=>{const target=location.hash&&document.getElementById(decodeURIComponent(location.hash.slice(1)));if(target&&reader.contains(target)){select('read');const detail=target.closest('details');if(detail)detail.open=true;}};
  addEventListener('hashchange',readHash);select('read');readHash();
})();
