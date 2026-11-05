"""Provider-assisted planning with repository context and bounded task steps."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Any,List
from .rag import RepositoryIndex
from .memory import MemoryStore
@dataclass
class PlanStep: id:int; title:str; action:str; status:str="pending"; result:Any=None
class ContextManager:
    def __init__(self,max_chars=24000): self.max_chars=max_chars; self.used=0
    def build(self,query,search_results,summary="",decisions=None):
        parts=["Task: "+query,"Summary: "+summary,"Decisions: "+str(decisions or [])]
        for r in search_results:
            block=f"\n[{r['path']}:{r['start_line']}-{r['end_line']}] {r['symbol']}\n{r['text']}"
            if sum(map(len,parts))+len(block)>self.max_chars: break
            parts.append(block)
        text="\n".join(parts); self.used=len(text); return text
class PlanningEngine:
    def __init__(self,root,provider=None,memory=None): self.root=root; self.provider=provider; self.rag=RepositoryIndex(root); self.memory=memory or MemoryStore(str(__import__('pathlib').Path(root)/'.minicoder_memory.sqlite')); self.context=ContextManager()
    def plan(self,request):
        self.rag.load() or self.rag.index(); hits=self.rag.search(request)
        context=self.context.build(request,hits,decisions=self.memory.search("decision"))
        if self.provider:
            response=self.provider.chat([{"role":"system","content":"Return a numbered coding plan only."},{"role":"user","content":context}]).text
            titles=[x.strip(" .") for x in response.splitlines() if x.strip()][:10]
        else: titles=["Analyze request and repository","Implement the requested changes","Run tests","Review failures and report"]
        return {"request":request,"context_chars":self.context.used,"steps":[asdict(PlanStep(i+1,t,t)) for i,t in enumerate(titles)],"evidence":hits}
