"""SQLite-backed scoped memory with secret redaction and update/forget operations."""
from __future__ import annotations
import json, re, sqlite3, time
from pathlib import Path
from typing import Any, Dict, List
SCOPES={"global","project","session","task"}
_SECRET=re.compile(r"(?i)(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*[^\s,;]+")
def redact(value):
    if isinstance(value,str): return _SECRET.sub(r"\1=[REDACTED]",value)
    if isinstance(value,dict): return {k:redact(v) for k,v in value.items()}
    if isinstance(value,list): return [redact(v) for v in value]
    return value
class MemoryStore:
    def __init__(self,path=".minicoder_memory.sqlite"):
        self.path=str(path); self.db=sqlite3.connect(self.path); self.db.execute("CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, scope TEXT, key TEXT, value TEXT, tags TEXT, updated REAL, UNIQUE(scope,key))"); self.db.commit()
    def put(self,scope,key,value,tags=None):
        if scope not in SCOPES: raise ValueError(f"scope must be one of {sorted(SCOPES)}")
        payload=json.dumps(redact(value),ensure_ascii=False); tag=json.dumps(tags or [],ensure_ascii=False)
        self.db.execute("INSERT INTO memories(scope,key,value,tags,updated) VALUES(?,?,?,?,?) ON CONFLICT(scope,key) DO UPDATE SET value=excluded.value,tags=excluded.tags,updated=excluded.updated",(scope,key,payload,tag,time.time())); self.db.commit()
    def get(self,scope,key):
        row=self.db.execute("SELECT value FROM memories WHERE scope=? AND key=?",(scope,key)).fetchone(); return json.loads(row[0]) if row else None
    def search(self,query,scope=None,limit=20):
        rows=self.db.execute("SELECT scope,key,value,tags,updated FROM memories WHERE scope=? OR ? IS NULL ORDER BY updated DESC",(scope,scope)).fetchall(); q=query.lower(); out=[]
        for s,k,v,t,u in rows:
            if q in (k+" "+v+" "+t).lower(): out.append({"scope":s,"key":k,"value":json.loads(v),"tags":json.loads(t),"updated":u})
        return out[:limit]
    def delete(self,scope,key): self.db.execute("DELETE FROM memories WHERE scope=? AND key=?",(scope,key)); self.db.commit()
    def close(self): self.db.close()
