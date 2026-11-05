"""Worker agent that registers with the controller and sends authenticated heartbeats."""
from __future__ import annotations
import os, platform, socket, threading, time
from typing import Optional
from urllib.parse import urlparse
import requests
import torch

class WorkerAgent:
    def __init__(self, controller_url: str, provider: str='local', auth_token: Optional[str]=None, timeout=10):
        parsed=urlparse(controller_url)
        if parsed.scheme not in {'http','https'} or not parsed.hostname:
            raise ValueError('controller_url must be an http(s) URL')
        if parsed.scheme=='http' and parsed.hostname not in {'127.0.0.1','::1','localhost'}:
            raise ValueError('HTTPS is required for non-loopback controller URLs')
        if timeout <= 0: raise ValueError('timeout must be positive')
        self.controller_url=controller_url.rstrip('/'); self.provider=provider; self.auth_token=auth_token or os.getenv('MINICODER_WORKER_TOKEN'); self.timeout=timeout
        self.worker_id=None; self.session=requests.Session(); self.stop_event=threading.Event(); self.gpu_info=self._detect_gpu(); self.registration=None
    def _headers(self): return {'Authorization':'Bearer '+self.auth_token} if self.auth_token else {}
    def _detect_gpu(self):
        if not torch.cuda.is_available(): return {'gpu_name':'CPU','vram_total':0,'vram_available':0,'cuda':False}
        prop=torch.cuda.get_device_properties(0); free,_=torch.cuda.mem_get_info()
        return {'gpu_name':prop.name,'vram_total':prop.total_memory//2**20,'vram_available':free//2**20,'cuda':True,'cuda_version':torch.version.cuda}
    def _payload(self):
        return {'worker_id':self.worker_id,'provider':self.provider,**self.gpu_info,'hostname':socket.gethostname(),'platform':platform.platform(),'python_version':platform.python_version(),'torch_version':torch.__version__,'network_id':os.getenv('MINICODER_NETWORK_ID'),'runtime':os.getenv('MINICODER_RUNTIME','python'),'distributed_backend':os.getenv('MINICODER_DISTRIBUTED_BACKEND','gloo' if not self.gpu_info['cuda'] else 'nccl'),'reachable':os.getenv('MINICODER_WORKER_REACHABLE','0')=='1','address':os.getenv('MINICODER_WORKER_ADDRESS',socket.gethostbyname(socket.gethostname())),'port':int(os.getenv('MINICODER_WORKER_PORT','0'))}
    def register(self)->bool:
        try:
            r=self.session.post(self.controller_url+'/workers/register',json=self._payload(),headers=self._headers(),timeout=self.timeout); r.raise_for_status(); data=r.json(); self.worker_id=data['worker_id']; self.registration=data; return True
        except (requests.RequestException,KeyError,ValueError) as e:
            print(f'Registration failed: {e}'); return False
    def send_heartbeat(self)->bool:
        if not self.worker_id: return False
        try:
            r=self.session.post(self.controller_url+'/workers/heartbeat',json=self._payload(),headers=self._headers(),timeout=self.timeout); r.raise_for_status(); return True
        except requests.RequestException as e: print(f'Heartbeat failed: {e}'); return False
    def run_heartbeat_loop(self,interval=10):
        if not self.worker_id and not self.register(): return False
        if interval <= 0: raise ValueError('interval must be positive')
        failures=0; delay=float(interval); max_failures=5
        while not self.stop_event.wait(delay):
            if self.send_heartbeat():
                failures=0; delay=float(interval)
            else:
                failures += 1
                if failures >= max_failures: return False
                delay=min(60.0, float(interval) * (2 ** failures))
        return True
    def stop(self): self.stop_event.set(); self.session.close()
if __name__=='__main__':
    agent=WorkerAgent(os.getenv('MINICODER_CONTROLLER_URL','http://127.0.0.1:8000'))
    try: agent.run_heartbeat_loop(float(os.getenv('MINICODER_HEARTBEAT_INTERVAL','10')))
    except KeyboardInterrupt: pass
    finally: agent.stop()
