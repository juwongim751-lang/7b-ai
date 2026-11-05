from typing import List
from collections import defaultdict
class GPUScheduler:
    """Only groups workers that can actually participate in one process group."""
    def __init__(self,registry): self.registry=registry
    def analyze_resources(self):
        workers=self.registry.get_active_workers(); return sum(w.vram_total for w in workers),len(workers),workers
    def compatible_groups(self):
        groups=defaultdict(list)
        for w in self.registry.get_active_workers():
            # Missing network identity/reachability is deliberately not treated as compatible.
            if not w.network_id or not w.reachable or w.runtime=='unknown' or w.distributed_backend=='unknown': continue
            key=(w.network_id,w.runtime,w.distributed_backend)
            groups[key].append(w)
        return [ws for ws in groups.values() if len(ws)>=2]
    def decide_training_mode(self,required_vram:int):
        _,_,workers=self.analyze_resources()
        if not workers: return 'NONE',[]
        for w in workers:
            if w.vram_available>=required_vram: return 'SINGLE_GPU',[w.worker_id]
        groups=self.compatible_groups()
        viable=[g for g in groups if sum(w.vram_available for w in g)>=required_vram]
        if viable:
            group=max(viable,key=lambda g:sum(w.vram_available for w in g)); return 'FSDP_DISTRIBUTED',[w.worker_id for w in group]
        return 'INDEPENDENT_TASKS',[w.worker_id for w in workers]
    def explain(self,required_vram):
        mode,ids=self.decide_training_mode(required_vram)
        return {'mode':mode,'workers':ids,'fsdp_groups':[[w.worker_id for w in g] for g in self.compatible_groups()], 'reason':'FSDP requires same network_id, runtime, backend, recent heartbeat and reachable=true; VRAM alone is insufficient'}
    def assign_ranks(self,worker_ids:List[str]): return {wid:{'rank':i,'world_size':len(worker_ids)} for i,wid in enumerate(worker_ids)}
