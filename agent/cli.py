from __future__ import annotations
import argparse,json
from .rag import RepositoryIndex, SentenceTransformerEmbeddingBackend
from .planner import PlanningEngine
from .quality import TestEngine,SelfReviewer

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--root",default="."); ap.add_argument("--embedding-model",default=None,help="enable true semantic search with sentence-transformers") ; sub=ap.add_subparsers(dest="cmd",required=True)
 sub.add_parser("index"); s=sub.add_parser("search"); s.add_argument("query"); s.add_argument("--limit",type=int,default=8)
 p=sub.add_parser("plan"); p.add_argument("request")
 sub.add_parser("test"); sub.add_parser("review")
 a=ap.parse_args(); root=a.root
 backend=SentenceTransformerEmbeddingBackend(a.embedding_model) if a.embedding_model else None
 if a.cmd in {"index","search"}: idx=RepositoryIndex(root,embedding_backend=backend); idx.load(); out=idx.index() if a.cmd=="index" else idx.search(a.query,a.limit)
 elif a.cmd=="plan": out=PlanningEngine(root).plan(a.request)
 elif a.cmd=="test": out=TestEngine(root).run()
 else: out=SelfReviewer(root).review()
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
