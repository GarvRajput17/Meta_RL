"""FastAPI wrapper for OpenSupplyChainEnv (Hugging Face Space deployment)."""

from __future__ import annotations

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from env.supply_chain_env import (
    Action,
    DepotAllocations,
    Observation,
    StepResult,
    SupplyChainEnv,
)

app = FastAPI(
    title="OpenSupplyChainEnv",
    version="1.0.0",
    description="Single-agent LLM-driven disaster resource allocation environment",
)

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
    return {"status": "healthy", "service": "open-supply-chain-env"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "open-supply-chain-env",
        "description": "Single-agent LLM-driven disaster resource allocation environment",
        "version": "1.0.0",
        "author": "Team QQuant",
        "tasks": [
            {"name": "static-baseline", "difficulty": "easy"},
            {"name": "demand-spike", "difficulty": "medium"},
            {"name": "cascading-failure", "difficulty": "hard"},
        ],
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "cdc_inventory": {"type": "integer"},
                "periodic_supply_rate": {"type": "integer"},
                "depot_inventories": {"type": "object"},
                "zone_demands": {"type": "object"},
                "road_status": {"type": "object"},
                "step": {"type": "integer"},
                "task_name": {"type": "string"},
                "episode_rewards": {"type": "array"},
            },
        },
    }


TASK_FILE_MAP: dict[str, str] = {
    "static-baseline": "task1",
    "demand-spike": "task2",
    "cascading-failure": "task3",
}

@app.post("/reset")
def reset(body: Optional[ResetRequest] = None) -> dict:
    req_body = body or ResetRequest()
    internal_task = TASK_FILE_MAP.get(req_body.task, req_body.task)
    try:
        obs = env.reset(internal_task)
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


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
