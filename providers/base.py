from abc import ABC, abstractmethod
import os

class BaseProvider(ABC):
    """
    현재 프로세스에서 관측 가능한 provider/runtime 상태를 살펴보는 인터페이스.
    setup_environment는 inspection 전용이며 인증, 설치, 원격 작업 제출을 하지 않는다.
    """
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    @abstractmethod
    def setup_environment(self):
        """로컬 환경 정보를 읽기 전용으로 조사한다."""
        pass

    @abstractmethod
    def get_storage_path(self) -> str:
        """체크포인트 및 데이터 저장을 위한 최적의 경로 반환"""
        pass

    @abstractmethod
    def run_worker_command(self, rank: int, world_size: int, master_addr: str, master_port: int):
        """해당 adapter가 지원한다고 명시한 경우에만 로컬용 명령 문자열을 반환한다."""
        pass
