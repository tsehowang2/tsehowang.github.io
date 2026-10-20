(() => {
  'use strict';
  if(window.COURSE_LANGUAGE!=='zh-HK')return;
  const dictionary=window.COURSE_TRANSLATIONS||{};
  const patterns=[
    [/^(\d+) matching pages( \(first 20\))?$/,(_,n,last)=>`找到 ${n} 個相符頁面${last?'（首 20 個）':''}`],
    [/^Semester (\d+) \/ Week (\d+)$/,(_,s,w)=>`第 ${s} 學期 / 第 ${w} 週`],
    [/^Lecture slides . Week (\d+)$/,(_,w)=>`課堂投影片 · 第 ${w} 週`],
    [/^Tutorial (\d+)$/,(_,n)=>`練習 ${n}`],
    [/^Week (\d+)$/,(_,n)=>`第 ${n} 週`],
    [/^(\d+) bus \+ (\d+) train \+ 1 bicycle = (\d+) matching sentences$/,(_,b,t,n)=>`${b} 個 bus + ${t} 個 train + 1 個 bicycle = ${n} 句相符句子`],
    ...(window.COURSE_LOCALE_PATTERNS||[]),
  ];
  function tr(s){
    if(Object.hasOwn(dictionary,s))return dictionary[s];
    const core=s.trim();
    if(Object.hasOwn(dictionary,core))return s.slice(0,s.length-s.trimStart().length)+dictionary[core]+s.slice(s.trimEnd().length);
    for(const [pattern,replacement] of patterns)if(pattern.test(core))return s.slice(0,s.length-s.trimStart().length)+core.replace(pattern,replacement)+s.slice(s.trimEnd().length);
    return s;
  }
  function localize(root){
    if(root.nodeType===Node.TEXT_NODE){
      if(root.parentElement?.closest('pre,code,script,style,textarea,[contenteditable],.language-switch'))return;
      const replacement=tr(root.nodeValue);if(replacement!==root.nodeValue)root.nodeValue=replacement;return;
    }
    if(root.nodeType!==Node.ELEMENT_NODE&&root!==document)return;
    if(root.nodeType===Node.ELEMENT_NODE){
      if(root.matches('pre,code,script,style,textarea,[contenteditable],.language-switch'))return;
      for(const attr of ['aria-label','placeholder','title','alt'])if(root.hasAttribute(attr)){
        const s=root.getAttribute(attr),replacement=tr(s);if(s!==replacement)root.setAttribute(attr,replacement);
      }
    }
    for(const child of root.childNodes)localize(child);
  }
  localize(document.body);
  const observer=new MutationObserver(records=>{
    for(const r of records){
      if(r.type==='characterData')localize(r.target);
      else if(r.type==='childList')for(const node of r.addedNodes)localize(node);
      else if(r.type==='attributes')localize(r.target);
    }
  });
  observer.observe(document.body,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['aria-label','placeholder','title','alt']});
})();
