"""FastAPI wrapper for OpenSupplyChainEnv (Hugging Face Space deployment)."""

from __future__ import annotations

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from env.supply_chain_env import (
    Action,
    DepotAllocations,
    SupplyChainEnv,
)

app = FastAPI(title="OpenSupplyChainEnv", version="1.0.0")

env = SupplyChainEnv(tasks_dir="tasks")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = "static-baseline"


class StepRequest(BaseModel):
    allocations: dict[str, int]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {"status": "ok", "env": "open-supply-chain-env"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": "open-supply-chain-env"}


@app.post("/reset")
def reset(body: Optional[ResetRequest] = None) -> dict:
    req_body = body or ResetRequest()
    try:
        obs = env.reset(req_body.task)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)})
    return obs.model_dump()


@app.post("/step")
def step(body: StepRequest) -> dict:
    try:
        action = Action(allocations=DepotAllocations(**body.allocations))
    except (ValidationError, TypeError) as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)})

    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state() -> dict:
    return env.state()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
