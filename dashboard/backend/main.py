from __future__ import annotations

import asyncio
import hmac
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from controller.registry import WorkerRegistry
from controller.scheduler import GPUScheduler
from dashboard.backend.control import TrainingProcessManager
from providers.registry import ProviderRegistry
from providers.scheduler import ProviderScheduler
from providers.kaggle_jobs import KaggleKernelClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = PROJECT_ROOT / "dashboard" / "frontend"
app = FastAPI(title="MiniCoder training dashboard")
registry = WorkerRegistry()
scheduler = GPUScheduler(registry)
manager = TrainingProcessManager(PROJECT_ROOT)
providers = ProviderRegistry()
provider_scheduler = ProviderScheduler(providers)
kaggle_client = KaggleKernelClient()
app.mount("/dashboard/static", StaticFiles(directory=str(FRONTEND)), name="dashboard-static")


@app.get("/")
def dashboard():
    return FileResponse(FRONTEND / "index.html")


class StartRequest(BaseModel):
    config: str = "config/model_tiny.yaml"
    train_config: Optional[str] = None
    run_dir: str = "runs/dashboard"
    resume: bool = False
    extra_args: list[str] = Field(default_factory=list)
    gpu_cost_rate_usd_per_hour: Optional[float] = None
    max_budget_usd: Optional[float] = None
    confirm_potential_cost: bool = False


class ExportRequest(BaseModel):
    output_dir: Optional[str] = None
    pack_bitnet: bool = False


class ProviderPlanRequest(BaseModel):
    required_vram_bytes: int = 0
    required_ram_bytes: int = 0
    required_storage_bytes: int = 0
    allow_paid: bool = False
    max_budget_usd: Optional[float] = None
    estimated_gpu_rate_usd_per_hour: Optional[float] = None


class KagglePrepareRequest(BaseModel):
    kernel_dir: str
    metadata: dict


class KaggleSubmitRequest(BaseModel):
    kernel_dir: str
    confirm_submit: bool = False


class KaggleKernelRequest(BaseModel):
    kernel_id: str


class KaggleOutputRequest(BaseModel):
    kernel_id: str
    destination_dir: str = "runs/kaggle_output"


def authorize_control(request: Request,
                      x_minicoder_token: Optional[str] = Header(default=None, alias="X-MiniCoder-Token")):
    """Mutating routes are loopback-only unless an explicit bearer token is configured."""
    expected = os.getenv("MINICODER_DASHBOARD_TOKEN")
    supplied = x_minicoder_token or request.headers.get("Authorization", "").removeprefix("Bearer ")
    peer = request.client.host if request.client else ""
    if expected:
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="valid dashboard token required")
    elif peer not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="remote control disabled; set MINICODER_DASHBOARD_TOKEN")


def authorize_worker(request: Request,
                     authorization: Optional[str] = Header(default=None, alias="Authorization"),
                     x_minicoder_token: Optional[str] = Header(default=None, alias="X-MiniCoder-Token")):
    expected = os.getenv("MINICODER_WORKER_TOKEN") or os.getenv("MINICODER_DASHBOARD_TOKEN")
    supplied = x_minicoder_token or (authorization.removeprefix("Bearer ") if authorization else "")
    peer = request.client.host if request.client else ""
    if expected:
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="valid worker token required")
    elif peer not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="remote worker registration disabled; configure a worker token")


def fail(error):
    raise HTTPException(status_code=409, detail=str(error))


@app.get("/workers", dependencies=[Depends(authorize_control)])
def get_workers():
    return registry.get_worker_status()


@app.post("/workers/register", dependencies=[Depends(authorize_worker)])
def register_worker(payload: dict):
    worker_id = registry.register(payload)
    return {"worker_id": worker_id, "status": "registered", "worker": registry.get_worker_status()[worker_id]}


@app.post("/workers/heartbeat", dependencies=[Depends(authorize_worker)])
def heartbeat_worker(payload: dict):
    worker_id = payload.get("worker_id")
    if not worker_id or not registry.update_heartbeat(worker_id, payload):
        raise HTTPException(status_code=404, detail="unknown worker_id; register first")
    return {"worker_id": worker_id, "status": "alive"}


@app.get("/status", dependencies=[Depends(authorize_control)])
def get_status():
    return manager.status()


@app.get("/providers", dependencies=[Depends(authorize_control)])
def get_providers():
    return providers.inspect_all()


@app.post("/providers/plan", dependencies=[Depends(authorize_control)])
def plan_provider(req: ProviderPlanRequest):
    values = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    return provider_scheduler.plan(**values)


@app.post("/providers/kaggle/prepare", dependencies=[Depends(authorize_control)])
def prepare_kaggle_kernel(req: KagglePrepareRequest):
    try:
        kernel_dir = manager._inside_root(req.kernel_dir, "Kaggle kernel directory", must_exist=True)
        return kaggle_client.prepare_kernel_metadata(kernel_dir, req.metadata)
    except Exception as error:
        fail(error)


@app.post("/providers/kaggle/submit", dependencies=[Depends(authorize_control)])
def submit_kaggle_kernel(req: KaggleSubmitRequest):
    try:
        kernel_dir = manager._inside_root(req.kernel_dir, "Kaggle kernel directory", must_exist=True)
        return kaggle_client.submit(kernel_dir, confirm_submit=req.confirm_submit)
    except Exception as error:
        fail(error)


@app.post("/providers/kaggle/status", dependencies=[Depends(authorize_control)])
def kaggle_kernel_status(req: KaggleKernelRequest):
    try:
        return kaggle_client.status(req.kernel_id)
    except Exception as error:
        fail(error)


@app.post("/providers/kaggle/output", dependencies=[Depends(authorize_control)])
def kaggle_kernel_output(req: KaggleOutputRequest):
    try:
        destination = manager._inside_root(req.destination_dir, "Kaggle output directory")
        return kaggle_client.download_output(req.kernel_id, destination)
    except Exception as error:
        fail(error)


@app.post("/train/start", dependencies=[Depends(authorize_control)])
def start_training(req: StartRequest = StartRequest()):
    try:
        return manager.start(req.config, req.train_config, req.run_dir, req.resume, req.extra_args,
                             req.gpu_cost_rate_usd_per_hour, req.max_budget_usd,
                             req.confirm_potential_cost)
    except Exception as error:
        fail(error)


@app.post("/train/stop", dependencies=[Depends(authorize_control)])
def stop_training():
    try:
        return manager.stop()
    except Exception as error:
        fail(error)


@app.post("/train/pause", dependencies=[Depends(authorize_control)])
def pause_training():
    try:
        return manager.pause()
    except Exception as error:
        fail(error)


@app.post("/train/resume", dependencies=[Depends(authorize_control)])
def resume_training():
    try:
        return manager.resume()
    except Exception as error:
        fail(error)


@app.post("/train/checkpoint", dependencies=[Depends(authorize_control)])
def checkpoint_training():
    try:
        return manager.checkpoint()
    except Exception as error:
        fail(error)


@app.post("/train/restore", dependencies=[Depends(authorize_control)])
def restore_training():
    try:
        return manager.restore()
    except Exception as error:
        fail(error)


@app.post("/train/export", dependencies=[Depends(authorize_control)])
def export_training(req: ExportRequest = ExportRequest()):
    try:
        return manager.export_model(req.output_dir, req.pack_bitnet)
    except Exception as error:
        fail(error)


@app.websocket("/ws/monitoring")
async def websocket_endpoint(websocket: WebSocket):
    peer = websocket.client.host if websocket.client else ""
    if peer not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({"workers": registry.get_worker_status(), "training": manager.status()})
            await asyncio.sleep(1)
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("MINICODER_DASHBOARD_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=int(os.getenv("MINICODER_DASHBOARD_PORT", "8000")))
