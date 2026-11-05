"""Repository index with AST chunks, lexical search and optional real embedding search."""
from __future__ import annotations
import ast, hashlib, json, math, re
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any, List, Optional
class EmbeddingBackend:
    name='custom'
    def encode(self,texts:List[str])->List[List[float]]: raise NotImplementedError
class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    name='sentence-transformers'
    def __init__(self,model='all-MiniLM-L6-v2',device=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e: raise RuntimeError('install sentence-transformers to enable true semantic RAG') from e
        self.model=SentenceTransformer(model,device=device)
    def encode(self,texts): return self.model.encode(texts,normalize_embeddings=True).tolist()
@dataclass
class CodeChunk:
    chunk_id:str; path:str; start_line:int; end_line:int; symbol:str; text:str; refs:List[str]
class RepositoryIndex:
    def __init__(self,root,index_dir=None,extensions=None,embedding_backend:Optional[EmbeddingBackend]=None):
        self.root=Path(root).resolve(); self.index_dir=Path(index_dir or self.root/'.minicoder_index'); self.index_dir.mkdir(parents=True,exist_ok=True)
        self.extensions=set(extensions or {'.py','.js','.ts','.java','.cpp','.c','.h','.cs','.go','.rs','.lua','.md','.txt'}); self.embedding_backend=embedding_backend; self.chunks=[]; self.files={}; self.embeddings={}
    def _files(self): return [p for p in self.root.rglob('*') if p.is_file() and p.suffix in self.extensions and '.minicoder_index' not in p.parts and '.git' not in p.parts]
    def _chunk_python(self,p,text):
        try: tree=ast.parse(text)
        except SyntaxError: return [CodeChunk(self._id(p,1,text),str(p.relative_to(self.root)),1,text.count('\n')+1,'<module>',text,[])]
        nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
        if not nodes: return [CodeChunk(self._id(p,1,text),str(p.relative_to(self.root)),1,text.count('\n')+1,'<module>',text,self._refs(text))]
        lines=text.splitlines(); out=[]
        for n in nodes:
            start,end=n.lineno,getattr(n,'end_lineno',n.lineno); body='\n'.join(lines[start-1:end]); out.append(CodeChunk(self._id(p,start,body),str(p.relative_to(self.root)),start,end,n.name,body,self._refs(body)))
        return out
    def _refs(self,text):
        pairs=re.findall(r'(?:from|import)\s+([A-Za-z_][\w.]*)|\b([A-Za-z_][\w]*)\s*\(',text); return sorted({a or b for a,b in pairs})
    def _id(self,p,line,text): return hashlib.sha1((str(p)+str(line)+text).encode()).hexdigest()[:16]
    def index(self,force=False):
        chunks=[]; changed=0; new_texts=[]; new_ids=[]
        for p in self._files():
            digest=hashlib.sha256(p.read_bytes()).hexdigest(); rel=str(p.relative_to(self.root)); old=self.files.get(rel)
            if not force and old and old.get('sha256')==digest:
                file_chunks=[CodeChunk(**x) for x in old['chunks']]; chunks+=file_chunks; continue
            text=p.read_text(encoding='utf-8',errors='replace'); changed+=1; file_chunks=self._chunk_python(p,text) if p.suffix=='.py' else [CodeChunk(self._id(p,1,text),rel,1,text.count('\n')+1,'<file>',text,self._refs(text))]
            chunks+=file_chunks; self.files[rel]={'sha256':digest,'chunks':[asdict(x) for x in file_chunks]}
            new_texts += [c.text for c in file_chunks]; new_ids += [c.chunk_id for c in file_chunks]
        self.chunks=chunks
        if self.embedding_backend and new_texts:
            self.embeddings.update({i:list(v) for i,v in zip(new_ids,self.embedding_backend.encode(new_texts))})
        self._save(); return {'files':len(self.files),'chunks':len(chunks),'reindexed':changed,'semantic_backend':self.embedding_backend.name if self.embedding_backend else None}
    def _save(self): (self.index_dir/'index.json').write_text(json.dumps({'root':str(self.root),'files':self.files,'embeddings':self.embeddings,'embedding_backend':self.embedding_backend.name if self.embedding_backend else None},indent=2),encoding='utf-8')
    def load(self):
        p=self.index_dir/'index.json'
        if not p.exists(): return False
        data=json.loads(p.read_text()); self.files=data.get('files',{}); self.embeddings=data.get('embeddings',{}); self.chunks=[CodeChunk(**x) for f in self.files.values() for x in f.get('chunks',[])]; return True
    @staticmethod
    def _tokens(s): return set(re.findall(r'[A-Za-z_][A-Za-z0-9_]+',s.lower()))
    @staticmethod
    def _cos(a,b):
        den=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b)); return sum(x*y for x,y in zip(a,b))/den if den else 0.0
    def search(self,query,limit=8,semantic=True):
        q=self._tokens(query); qvec=self.embedding_backend.encode([query])[0] if semantic and self.embedding_backend else None; scored=[]
        for c in self.chunks:
            t=self._tokens(c.text+' '+c.symbol+' '+c.path); lexical=len(q&t)/(math.sqrt(max(1,len(q))*max(1,len(t)))); lexical += 1.0 if query.lower() in c.text.lower() else 0.0
            sem=self._cos(qvec,self.embeddings.get(c.chunk_id,[])) if qvec is not None else 0.0
            score=(0.35*lexical+0.65*sem) if qvec is not None else lexical
            if score: scored.append((score,c,sem))
        return [{**asdict(c),'score':round(s,6),'semantic_score':round(sem,6),'search_mode':'semantic+lexical' if qvec is not None else 'lexical-only'} for s,c,sem in sorted(scored,key=lambda x:(-x[0],x[1].path,x[1].start_line))[:limit]]
