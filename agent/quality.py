"""Bounded shell-free execution, test orchestration and deterministic self-review checks."""
from __future__ import annotations
import ast, re, shlex, subprocess, sys
from dataclasses import dataclass,asdict
from pathlib import Path
@dataclass
class RunResult: command:str; ok:bool; stdout:str=''; stderr:str=''; returncode:int|None=None; timed_out:bool=False
class CodeExecutor:
    def __init__(self,root='.',timeout=30): self.root=Path(root).resolve(); self.timeout=timeout
    def run(self,command):
        argv=list(command) if isinstance(command,(list,tuple)) else shlex.split(command)
        if not argv: return RunResult(str(command),False,stderr='empty command')
        try:
            p=subprocess.run(argv,shell=False,cwd=self.root,text=True,capture_output=True,timeout=self.timeout,start_new_session=True); return RunResult(str(command),p.returncode==0,p.stdout,p.stderr,p.returncode)
        except subprocess.TimeoutExpired as e: return RunResult(str(command),False,e.stdout or '',f'timeout after {self.timeout}s',None,True)
        except OSError as e: return RunResult(str(command),False,'',str(e),None)
    def python(self,path,*args): return self.run([sys.executable,str(path),*map(str,args)])
    def test(self,command=None): return self.run(command or [sys.executable,'-m','unittest','discover','-v'])
class TestEngine:
    def __init__(self,root='.',max_retries=1,timeout=60): self.root=Path(root); self.max_retries=max_retries; self.executor=CodeExecutor(root,timeout)
    def run(self,commands=None):
        commands=commands or [[sys.executable,'-m','compileall','-q','.'],[sys.executable,'-m','unittest','discover','-v']]; results=[]
        for cmd in commands:
            for attempt in range(self.max_retries+1):
                last=self.executor.run(cmd)
                if last.ok: break
            results.append({'attempts':attempt+1,**asdict(last)})
        return {'passed':all(x['ok'] for x in results),'executed':True,'results':results}
class SelfReviewer:
    def __init__(self,root='.'): self.root=Path(root)
    def review(self):
        findings=[]
        for p in self.root.rglob('*.py'):
            if '.git' in p.parts or '__pycache__' in p.parts: continue
            try: ast.parse(p.read_text(encoding='utf-8'))
            except SyntaxError as e: findings.append({'severity':'error','path':str(p),'issue':f'syntax: {e}'})
            text=p.read_text(encoding='utf-8',errors='replace')
            if re.search(r"(?:password|api_key|secret)\s*=\s*['\"][^'\"]+",text,re.I): findings.append({'severity':'warning','path':str(p),'issue':'possible hard-coded secret'})
            if 'subprocess.run(' in text and 'timeout=' not in text: findings.append({'severity':'warning','path':str(p),'issue':'subprocess without timeout'})
            shell_flag='shell'+'=True'
            if shell_flag in text: findings.append({'severity':'error','path':str(p),'issue':'shell execution is forbidden'})
        return {'passed':not any(x['severity']=='error' for x in findings),'findings':findings}
