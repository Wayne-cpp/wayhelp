"""后台作业 API(/api/jobs/*):白名单作业名;运行中重复触发 409。"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/jobs")


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code,
                                                               "message": message}})


@router.post("/{name}/run")
async def job_run(name: str, request: Request):
    runner = request.app.state.job_runner
    if name not in runner.names():
        return _err(404, "job_unknown", "未知作业")
    if not await runner.run(name):
        return _err(409, "job_running", "作业正在运行")
    return {"started": True}


@router.get("/{name}")
async def job_status(name: str, request: Request):
    runner = request.app.state.job_runner
    if name not in runner.names():
        return _err(404, "job_unknown", "未知作业")
    return runner.status(name)
