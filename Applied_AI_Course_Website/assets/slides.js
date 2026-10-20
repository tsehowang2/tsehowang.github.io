(() => {
  'use strict';
  document.querySelectorAll('.lecture-deck').forEach(deck => {
    const slides=[...deck.querySelectorAll('[data-slide]')];
    const prev=deck.querySelector('.slide-prev'),next=deck.querySelector('.slide-next');
    const counter=deck.querySelector('.slide-counter'),jump=deck.querySelector('.slide-jump');
    const viewport=deck.querySelector('.slide-viewport'),present=deck.querySelector('.deck-present');
    const all=deck.querySelector('.deck-all');let index=0,showAll=false;
    function render(){
      slides.forEach((slide,i)=>slide.hidden=!showAll && i!==index);
      counter.textContent=`${index+1} / ${slides.length}`;
      prev.disabled=index===0;next.disabled=index===slides.length-1;jump.value=String(index);
      deck.style.setProperty('--slide-progress',`${(index+1)/slides.length*100}%`);
    }
    function move(delta){index=Math.max(0,Math.min(slides.length-1,index+delta));render();if(deck.classList.contains('presenting'))deck.scrollTop=0;}
    prev.addEventListener('click',()=>move(-1));next.addEventListener('click',()=>move(1));
    jump.addEventListener('change',()=>{index=Number(jump.value);render();});
    deck.addEventListener('keydown',e=>{
      if(e.target.matches('input,textarea,select'))return;
      if(e.key==='ArrowRight'){e.preventDefault();move(1);}
      if(e.key==='ArrowLeft'){e.preventDefault();move(-1);}
      if(e.key==='Escape'&&deck.classList.contains('presenting'))togglePresent();
    });
    function togglePresent(){
      const active=deck.classList.toggle('presenting');
      document.body.classList.toggle('slide-presentation-open',active);
      present.textContent=active?'Exit presentation (Esc)':'Present';
      present.setAttribute('aria-pressed',String(active));
      if(active){deck.scrollTop=0;viewport.focus({preventScroll:true});}else present.focus({preventScroll:true});
    }
    present.addEventListener('click',togglePresent);
    all.addEventListener('click',()=>{showAll=!showAll;deck.classList.toggle('all-slides',showAll);all.textContent=showAll?'Show one slide':'Show all slides';all.setAttribute('aria-pressed',String(showAll));render();});
    render();
  });
})();
