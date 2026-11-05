"""Agent tools with shell-free argv execution and explicit approval boundaries."""
from __future__ import annotations
import ast, os, shlex, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
@dataclass
class ToolResult:
    ok:bool; output:str=''; error:str=''; returncode:int|None=None
class ToolRegistry:
    def __init__(self,root='.',timeout=30,allowed_commands=None,max_output=1_000_000):
        self.root=Path(root).resolve(); self.timeout=timeout; self.max_output=max_output
        self.allowed_commands=set(allowed_commands or {'python','python3','git','pytest','mypy','ruff','ls','pwd','find'})
    def _path(self,p):
        x=(self.root/p).resolve()
        if self.root not in x.parents and x!=self.root: raise ValueError('path escapes repository root')
        return x
    def read_file(self,path): return ToolResult(True,self._path(path).read_text(encoding='utf-8',errors='replace'))
    def write_file(self,path,content,approved=False):
        if not approved: return ToolResult(False,error='approval required for file creation/modification')
        p=self._path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(content,encoding='utf-8'); return ToolResult(True,'written '+str(path))
    def search_files(self,pattern): return ToolResult(True,'\n'.join(str(p.relative_to(self.root)) for p in self.root.rglob(pattern) if p.is_file()))
    def list_dir(self,path='.'): return ToolResult(True,'\n'.join(sorted(x.name for x in self._path(path).iterdir())))
    def _argv(self,command):
        argv=list(command) if isinstance(command,(list,tuple)) else shlex.split(command,posix=os.name!='nt')
        if not argv: raise ValueError('empty command')
        exe=Path(argv[0]).name.lower()
        if exe not in self.allowed_commands: raise PermissionError(f'command not allowlisted: {exe}')
        return argv
    def run_command(self,command,approved=False):
        try: argv=self._argv(command)
        except (ValueError,PermissionError) as e: return ToolResult(False,error=str(e))
        exe=Path(argv[0]).name.lower(); privileged=exe in {'git','python','python3','pytest','mypy','ruff'} and any(x in argv for x in ('-c','push','commit','reset','clean','install'))
        if privileged and not approved: return ToolResult(False,error='approval required for code-changing or repository-changing command')
        try:
            p=subprocess.run(argv,shell=False,cwd=self.root,text=True,capture_output=True,timeout=self.timeout,start_new_session=(os.name!='nt'))
            out=(p.stdout or '')[:self.max_output]; err=(p.stderr or '')[:self.max_output]
            return ToolResult(p.returncode==0,out,err,p.returncode)
        except subprocess.TimeoutExpired as e: return ToolResult(False,(e.stdout or '')[:self.max_output],f'timeout after {self.timeout}s')
        except OSError as e: return ToolResult(False,error=str(e))
    def run_python(self,code,approved=False): return self.run_command([sys.executable,'-c',code],approved=approved)
    def syntax_check(self,path):
        try: ast.parse(self._path(path).read_text(encoding='utf-8')); return ToolResult(True,'syntax ok')
        except (SyntaxError,OSError) as e: return ToolResult(False,error=str(e))
    def git(self,subcommand,approved=False): return self.run_command(['git',*shlex.split(subcommand)],approved=approved)
    def schema(self): return {'read_file':{'path':'str'},'write_file':{'path':'str','content':'str','approved':'bool'},'search_files':{'pattern':'str'},'run_command':{'argv':'list[str]','approved':'bool'},'run_python':{'code':'str','approved':'bool'},'syntax_check':{'path':'str'},'git':{'subcommand':'str','approved':'bool'}}
