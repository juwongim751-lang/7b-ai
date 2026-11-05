import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.distributed as dist
import os

class DistributedManager:
    """
    FSDP(Fully Sharded Data Parallel)를 사용하여 7B 모델을 분산 학습 환경에 배치하는 매니저.
    """
    def __init__(self, config):
        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    def setup(self):
        """분산 환경 초기화"""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

    def cleanup(self):
        """분산 환경 종료"""
        dist.destroy_process_group()

    def wrap_model(self, model):
        """
        모델을 FSDP로 래핑하여 파라미터, 그래디언트, 옵티마이저 상태를 분할(Sharding)합니다.
        """
        # 1. Mixed Precision 설정 (BF16 권장)
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16,
            reduce_dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16,
            buffer_dtype=torch.float32,
        )

        # 2. Auto-wrap 정책: 특정 크기 이상의 레이어는 자동으로 쪼개어 관리
        # MiniCoderBlock과 같은 Transformer 레이어 단위로 래핑
        auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1e6)

        # 3. FSDP 래핑
        # ShardingStrategy.FULL_SHARD: 파라미터, 그래디언트, 옵티마이저 상태 모두 분할 (메모리 최소화)
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device()
        )
        
        return fsdp_model

    def get_distributed_optimizer(self, model, lr=3e-4):
        """FSDP 모델에 최적화된 Optimizer 생성"""
        # FSDP 모델의 경우 parameters() 호출 시 shard된 파라미터만 반환됨
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        return optimizer

if __name__ == "__main__":
    # FSDP 동작 테스트 (단일 노드 다중 GPU 가정)
    # 실행 명령어 예시: torchrun --nproc_per_node=2 training/distributed.py
    from model.config import MiniCoderConfig
    from model.transformer import MiniCoder7B
    
    cfg = MiniCoderConfig()
    mgr = DistributedManager(cfg)
    mgr.setup()
    
    model = MiniCoder7B(cfg).to(mgr.local_rank)
    fsdp_model = mgr.wrap_model(model)
    
    print(f"Rank {mgr.rank}: Model wrapped with FSDP. Total parameters sharded across {mgr.world_size} GPUs.")
    
    mgr.cleanup()
