"""Comprehensive tests for OpenSupplyChainEnv."""

from __future__ import annotations

# HF_TOKEN must be set before importing inference (module-level check).
import os

os.environ.setdefault("HF_TOKEN", "test-token")

import io
import json
import re
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from env.supply_chain_env import (
    Action,
    DEPOT_CAPACITY,
    DEPOT_TO_ZONES,
    DEPOTS,
    DepotAllocations,
    EDGES,
    MAX_STEPS,
    Observation,
    StepResult,
    SupplyChainEnv,
    VALID_ALLOC,
    ZONES,
)
from graders.graders import (
    compute_normalised_score,
    proportional_heuristic_action,
    run_heuristic_grader,
)

ZERO = Action(allocations=DepotAllocations(depotA=0, depotB=0, depotC=0))


# =========================================================================
# Helpers
# =========================================================================

def _write_task(
    tmp_path: Path,
    name: str,
    schedule: list,
    *,
    cdc: int = 2000,
    periodic: int = 0,
) -> str:
    """Write a minimal valid task JSON into *tmp_path* and return the dir."""
    data = {
        "task_name": name,
        "cdc_initial_inventory": cdc,
        "periodic_supply_rate": periodic,
        "depot_initial_inventories": {"depotA": 200, "depotB": 200, "depotC": 200},
        "base_zone_demands": {f"zone{i}": 50 for i in range(1, 7)},
        "disruption_schedule": schedule,
    }
    (tmp_path / f"{name}.json").write_text(json.dumps(data))
    return str(tmp_path)


# =========================================================================
# 1. reset() returns valid Observation for all 3 tasks
# =========================================================================

_TASK_INITIAL_CDC = {"task1": 1500, "task2": 1400, "task3": 1200}


@pytest.mark.parametrize("task", ["task1", "task2", "task3"])
def test_reset_returns_observation(task: str) -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    obs = env.reset(task)
    assert isinstance(obs, Observation)
    assert obs.step == 0
    assert obs.cdc_inventory == _TASK_INITIAL_CDC[task]
    assert set(obs.depot_inventories.keys()) == set(DEPOTS)
    assert set(obs.zone_demands.keys()) == set(ZONES)
    assert set(obs.road_status.keys()) == set(EDGES)
    assert isinstance(obs.pending_resupplies, list)


# =========================================================================
# 2. valid step() returns StepResult with sane reward
# =========================================================================

def test_valid_step_returns_step_result() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task1")
    act = Action(allocations=DepotAllocations(depotA=100, depotB=100, depotC=100))
    result = env.step(act)
    assert isinstance(result, StepResult)
    assert -10.0 <= result.reward <= 1.0


# =========================================================================
# 3. invalid action over CDC returns reward=-5.0 with error
# =========================================================================

def test_invalid_action_exceeds_cdc(tmp_path: Path) -> None:
    tasks_dir = _write_task(tmp_path, "drain", [], cdc=2000, periodic=0)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    env.reset("drain")
    # Step 0: (400,400,400)=1200, CDC 2000->800, depots 200+400=600->500 after distrib
    env.step(Action(allocations=DepotAllocations(depotA=400, depotB=400, depotC=400)))
    # Step 1: headroom=100, (100,100,100)=300, CDC 800->500
    env.step(Action(allocations=DepotAllocations(depotA=100, depotB=100, depotC=100)))
    # Step 2: headroom=100, (100,100,100)=300, CDC 500->200
    env.step(Action(allocations=DepotAllocations(depotA=100, depotB=100, depotC=100)))
    # Step 3: (100,100,100)=300 > CDC=200 -> INVALID
    result = env.step(Action(allocations=DepotAllocations(depotA=100, depotB=100, depotC=100)))
    assert result.reward == -5.0
    assert result.info["error"] is not None


# =========================================================================
# 4. invalid action must not mutate state
# =========================================================================

def test_invalid_action_no_state_mutation(tmp_path: Path) -> None:
    tasks_dir = _write_task(tmp_path, "drain", [], cdc=2000, periodic=0)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    env.reset("drain")
    env.step(Action(allocations=DepotAllocations(depotA=400, depotB=400, depotC=400)))
    env.step(Action(allocations=DepotAllocations(depotA=100, depotB=100, depotC=100)))
    env.step(Action(allocations=DepotAllocations(depotA=100, depotB=100, depotC=100)))
    before = env.state()
    # CDC is 200; try 300 -> invalid
    env.step(Action(allocations=DepotAllocations(depotA=100, depotB=100, depotC=100)))
    after = env.state()
    assert before["cdc_inventory"] == after["cdc_inventory"]
    assert before["depot_inventories"] == after["depot_inventories"]
    assert before["zone_demands"] == after["zone_demands"]
    assert before["road_status"] == after["road_status"]
    assert len(before["episode_rewards"]) == len(after["episode_rewards"])


# =========================================================================
# 5. invalid action must not advance step
# =========================================================================

def test_invalid_action_no_step_advance(tmp_path: Path) -> None:
    tasks_dir = _write_task(tmp_path, "drain", [], cdc=2000, periodic=0)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    env.reset("drain")
    env.step(Action(allocations=DepotAllocations(depotA=400, depotB=400, depotC=400)))
    env.step(Action(allocations=DepotAllocations(depotA=100, depotB=100, depotC=100)))
    env.step(Action(allocations=DepotAllocations(depotA=100, depotB=100, depotC=100)))
    step_before = env.state()["step"]
    # CDC is 200; try 300 -> invalid, step must not advance
    env.step(Action(allocations=DepotAllocations(depotA=100, depotB=100, depotC=100)))
    assert env.state()["step"] == step_before


# =========================================================================
# 6. invalid allocation values rejected by Pydantic
# =========================================================================

def test_pydantic_rejects_invalid_value() -> None:
    with pytest.raises(Exception):
        DepotAllocations(depotA=99, depotB=0, depotC=0)


# =========================================================================
# 7. extra keys rejected by Pydantic
# =========================================================================

def test_pydantic_rejects_extra_keys() -> None:
    with pytest.raises(Exception):
        DepotAllocations(depotA=100, depotB=100, depotC=100, depotD=0)


# =========================================================================
# 8. missing keys rejected by Pydantic
# =========================================================================

def test_pydantic_rejects_missing_keys() -> None:
    with pytest.raises(Exception):
        DepotAllocations(depotA=100, depotB=100)


# =========================================================================
# 9. task3 step 3 closes CDC->depotB
# =========================================================================

def test_task3_road_closure_step3() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task3")
    for _ in range(3):
        env.step(ZERO)
    assert env.state()["road_status"]["CDC->depotB"] == "closed"


# =========================================================================
# 10. task2 step 4 sets zone3 demand to 160
# =========================================================================

def test_task2_demand_spike_step4() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task2")
    for _ in range(4):
        env.step(ZERO)
    assert env.state()["zone_demands"]["zone3"] == 160


# =========================================================================
# 11. task2 step 12 resets zone3 demand to 70
# =========================================================================

def test_task2_demand_reset_step12() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task2")
    for _ in range(12):
        env.step(ZERO)
    assert env.state()["zone_demands"]["zone3"] == 70


# =========================================================================
# 11b. task2 step 6 closes depotA->zone2 road
# =========================================================================

def test_task2_zone_road_closure_step6() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task2")
    for _ in range(6):
        env.step(ZERO)
    assert env.state()["road_status"]["depotA->zone2"] == "closed"


# =========================================================================
# 11c. task2 step 14 reopens depotA->zone2 road
# =========================================================================

def test_task2_zone_road_reopen_step14() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task2")
    for _ in range(14):
        env.step(ZERO)
    assert env.state()["road_status"]["depotA->zone2"] == "open"


# =========================================================================
# 12. task3 step 11 reopens CDC->depotB
# =========================================================================

def test_task3_road_reopen_step11() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task3")
    for _ in range(11):
        env.step(ZERO)
    assert env.state()["road_status"]["CDC->depotB"] == "open"


# =========================================================================
# 13. same-step conflict: road_open wins over road_closure
# =========================================================================

def test_same_step_road_open_wins(tmp_path: Path) -> None:
    schedule = [
        {"type": "road_closure", "step": 0, "edge": "CDC->depotA"},
        {"type": "road_open", "step": 0, "edge": "CDC->depotA"},
    ]
    tasks_dir = _write_task(tmp_path, "conflict", schedule)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    obs = env.reset("conflict")
    assert obs.road_status["CDC->depotA"] == "open"


# =========================================================================
# 14. same-step multiple demand_change: last one wins
# =========================================================================

def test_same_step_demand_change_last_wins(tmp_path: Path) -> None:
    schedule = [
        {"type": "demand_change", "step": 0, "zone": "zone1", "units": 999},
        {"type": "demand_change", "step": 0, "zone": "zone1", "units": 42},
    ]
    tasks_dir = _write_task(tmp_path, "demdup", schedule)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    obs = env.reset("demdup")
    assert obs.zone_demands["zone1"] == 42


# =========================================================================
# 15. same-step multiple cdc_inventory_cut: factors compound
# =========================================================================

def test_same_step_cdc_cuts_compound(tmp_path: Path) -> None:
    schedule = [
        {"type": "cdc_inventory_cut", "step": 0, "factor": 0.5},
        {"type": "cdc_inventory_cut", "step": 0, "factor": 0.5},
    ]
    tasks_dir = _write_task(tmp_path, "cutdup", schedule)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    obs = env.reset("cutdup")
    # 2000 * 0.5 = 1000, then int(1000 * 0.5) = 500
    assert obs.cdc_inventory == 500


# =========================================================================
# 15b. cdc_resupply: Pydantic model validates correctly
# =========================================================================

def test_cdc_resupply_event_model() -> None:
    from env.supply_chain_env import CdcResupplyEvent
    ev = CdcResupplyEvent(type="cdc_resupply", step=5, units=300)
    assert ev.units == 300
    assert ev.step == 5
    with pytest.raises(Exception):
        CdcResupplyEvent(type="cdc_resupply", step=5, units=300, extra_field=1)


# =========================================================================
# 15c. cdc_resupply applies correctly via _apply_disruptions
# =========================================================================

def test_cdc_resupply_applies(tmp_path: Path) -> None:
    schedule = [
        {"type": "cdc_resupply", "step": 1, "units": 500},
    ]
    tasks_dir = _write_task(tmp_path, "resupply", schedule, cdc=2000, periodic=0)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    env.reset("resupply")
    result = env.step(ZERO)
    assert result.observation.cdc_inventory == 2000 + 500


# =========================================================================
# 15d. task2 step 20: cdc_resupply (with periodic supply)
# =========================================================================

def test_task2_cdc_resupply_step20() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task2")
    for _ in range(20):
        env.step(ZERO)
    # CDC=1400, periodic=170 at steps 1-20 (+3400), resupply +500 at step 10, +400 at step 20
    assert env.state()["cdc_inventory"] == 1400 + 170 * 20 + 500 + 400  # 5700


# =========================================================================
# 15e. task3 step 18: cdc_resupply (with periodic + cut)
# =========================================================================

def test_task3_cdc_resupply_step18() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task3")
    for _ in range(18):
        env.step(ZERO)
    # CDC=1200, periodic=200/step.
    # Steps 1-7: +200*7=1400 → 2600
    # Step 8: +200=2800, cut ×0.5 → int(2800*0.5)=1400
    # Steps 9-14: +200*6=1200 → 2600
    # Step 15: +200=2800, resupply +500 → 3300
    # Steps 16-17: +200*2=400 → 3700
    # Step 18: +200=3900 (CDC->depotC closes, no CDC effect with ZERO)
    assert env.state()["cdc_inventory"] == 3900


# =========================================================================
# 15f. same-step: cut then resupply (Phase 4 before Phase 5)
# =========================================================================

def test_same_step_cut_then_resupply(tmp_path: Path) -> None:
    schedule = [
        {"type": "cdc_inventory_cut", "step": 0, "factor": 0.5},
        {"type": "cdc_resupply", "step": 0, "units": 200},
    ]
    tasks_dir = _write_task(tmp_path, "cutresup", schedule, cdc=2000, periodic=0)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    obs = env.reset("cutresup")
    # 2000 * 0.5 = 1000, then +200 = 1200
    assert obs.cdc_inventory == 1200


# =========================================================================
# 15g. validate_task_file rejects invalid cdc_resupply units
# =========================================================================

def test_validator_rejects_bad_resupply_units(tmp_path: Path) -> None:
    data = {
        "task_name": "bad-resupply",
        "cdc_initial_inventory": 2000,
        "periodic_supply_rate": 0,
        "depot_initial_inventories": {"depotA": 200, "depotB": 200, "depotC": 200},
        "base_zone_demands": {f"zone{i}": 50 for i in range(1, 7)},
        "disruption_schedule": [
            {"type": "cdc_resupply", "step": 1, "units": -100},
        ],
    }
    p = tmp_path / "bad-resupply.json"
    p.write_text(json.dumps(data))
    from env.supply_chain_env import validate_task_file
    with pytest.raises(ValueError, match="positive int"):
        validate_task_file(str(p))


# =========================================================================
# 15h. periodic_supply_rate adds to CDC each step from step 1 onward
# =========================================================================

def test_periodic_supply_fires_each_step(tmp_path: Path) -> None:
    tasks_dir = _write_task(tmp_path, "periodic", [], cdc=1000, periodic=100)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    obs = env.reset("periodic")
    assert obs.cdc_inventory == 1000  # no periodic at step 0
    r1 = env.step(ZERO)
    assert r1.observation.cdc_inventory == 1100  # step 1: +100
    r2 = env.step(ZERO)
    assert r2.observation.cdc_inventory == 1200  # step 2: +100


# =========================================================================
# 15i. periodic supply interacts correctly with cuts (Phase 0 before Phase 4)
# =========================================================================

def test_periodic_then_cut(tmp_path: Path) -> None:
    schedule = [
        {"type": "cdc_inventory_cut", "step": 1, "factor": 0.5},
    ]
    tasks_dir = _write_task(tmp_path, "pcut", schedule, cdc=1000, periodic=200)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    env.reset("pcut")
    result = env.step(ZERO)
    # Step 1: Phase 0 periodic +200 -> 1200, then Phase 4 cut *0.5 -> 600
    assert result.observation.cdc_inventory == 600


# =========================================================================
# 15j. pending_resupplies only shows future events
# =========================================================================

def test_pending_resupplies_future_only(tmp_path: Path) -> None:
    schedule = [
        {"type": "cdc_resupply", "step": 1, "units": 100},
        {"type": "cdc_resupply", "step": 5, "units": 200},
    ]
    tasks_dir = _write_task(tmp_path, "pendtest", schedule, cdc=1000, periodic=0)
    env = SupplyChainEnv(tasks_dir=tasks_dir)
    obs = env.reset("pendtest")
    assert len(obs.pending_resupplies) == 2  # both at step 1 and 5 are future
    r = env.step(ZERO)
    # After step 0 → advance to step 1 → resupply at step 1 applied
    # Pending should only show step 5
    assert len(r.observation.pending_resupplies) == 1
    assert r.observation.pending_resupplies[0]["step"] == 5


# =========================================================================
# 15k. task3 step 2 closes depotA->zone1
# =========================================================================

def test_task3_zone_road_closure_step2() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task3")
    for _ in range(2):
        env.step(ZERO)
    assert env.state()["road_status"]["depotA->zone1"] == "closed"


# =========================================================================
# 15l. task3 step 10 reopens depotA->zone1
# =========================================================================

def test_task3_zone_road_reopen_step10() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task3")
    for _ in range(10):
        env.step(ZERO)
    assert env.state()["road_status"]["depotA->zone1"] == "open"


# =========================================================================
# 15m. task3 step 18 closes CDC->depotC
# =========================================================================

def test_task3_depotc_closure_step18() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task3")
    for _ in range(18):
        env.step(ZERO)
    assert env.state()["road_status"]["CDC->depotC"] == "closed"


# =========================================================================
# 15n. task3 step 22 reopens CDC->depotC
# =========================================================================

def test_task3_depotc_reopen_step22() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task3")
    for _ in range(22):
        env.step(ZERO)
    assert env.state()["road_status"]["CDC->depotC"] == "open"


# =========================================================================
# 15o. task3 double CDC cuts at step 8 and step 25
# =========================================================================

def test_task3_double_cdc_cut() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task3")
    for _ in range(25):
        env.step(ZERO)
    # CDC=1200, periodic=200/step.
    # Steps 1-7: 1200+1400=2600. Step 8: +200=2800, cut×0.5=1400.
    # Steps 9-14: +200*6=1200 → 2600. Step 15: +200+500=3300.
    # Steps 16-21: +200*6=1200 → 4500.
    # Step 22: +200=4700, resupply+400=5100 (depotC reopens).
    # Steps 23-24: +200*2=400 → 5500.
    # Step 25: +200=5700, cut×0.7=int(5700*0.7)=3989 (float rounding).
    assert env.state()["cdc_inventory"] == 3989


# =========================================================================
# 16. full episode runs 30 steps for all tasks
# =========================================================================

@pytest.mark.parametrize("task", ["task1", "task2", "task3"])
def test_full_episode_30_steps(task: str) -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset(task)
    result = None
    for _ in range(MAX_STEPS):
        result = env.step(ZERO)
    assert result is not None
    assert result.done is True
    assert result.observation.step == MAX_STEPS
    assert len(env.state()["episode_rewards"]) == MAX_STEPS


# =========================================================================
# 17. state() returns a deep copy
# =========================================================================

def test_state_deep_copy() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    env.reset("task1")
    s1 = env.state()
    s1["depot_inventories"]["depotA"] = 9999
    s1["episode_rewards"].append(99.0)
    s2 = env.state()
    assert s2["depot_inventories"]["depotA"] == 200
    assert len(s2["episode_rewards"]) == 0


# =========================================================================
# 18. Observation from reset() is safe to mutate
# =========================================================================

def test_observation_mutation_safe() -> None:
    env = SupplyChainEnv(tasks_dir="tasks")
    obs = env.reset("task1")
    obs.depot_inventories["depotA"] = 9999
    assert env.state()["depot_inventories"]["depotA"] == 200


# =========================================================================
# 19. run_heuristic_grader returns normalised_score in [0,1]
# =========================================================================

@pytest.mark.parametrize("task", ["task1", "task2", "task3"])
def test_heuristic_grader_score_range(task: str) -> None:
    result = run_heuristic_grader(task)
    assert 0.0 <= result["normalised_score"] <= 1.0
    assert result["policy"] == "proportional_heuristic"
    assert len(result["per_step_rewards"]) == MAX_STEPS


# =========================================================================
# 20–22. compute_normalised_score edge cases
# =========================================================================

def test_normalised_score_zero() -> None:
    assert compute_normalised_score(0) == 1.0


def test_normalised_score_minus_five() -> None:
    assert compute_normalised_score(-5) == 0.0


def test_normalised_score_minus_two_point_five() -> None:
    assert compute_normalised_score(-2.5) == 0.5


# =========================================================================
# 23. grader is deterministic across repeated calls
# =========================================================================

def test_grader_deterministic() -> None:
    r1 = run_heuristic_grader("task1")
    r2 = run_heuristic_grader("task1")
    assert r1["total_reward"] == r2["total_reward"]
    assert r1["per_step_rewards"] == r2["per_step_rewards"]


# =========================================================================
# 24. openenv.yaml has expected tasks and api keys
# =========================================================================

def test_openenv_yaml() -> None:
    path = Path("openenv.yaml")
    assert path.exists()
    cfg = yaml.safe_load(path.read_text())
    assert cfg["name"] == "open-supply-chain-env"
    task_names = [t["name"] for t in cfg["tasks"]]
    assert task_names == ["static-baseline", "demand-spike", "cascading-failure"]
    assert "reset" in cfg["api"]
    assert "step" in cfg["api"]
    assert "state" in cfg["api"]
    assert cfg["entry_point"] == "inference.py"
    assert "openenv" in cfg["tags"]


# =========================================================================
# 25–30. FastAPI route tests
# =========================================================================

@pytest.fixture()
def api_client():
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


def test_fastapi_health(api_client) -> None:
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "env": "open-supply-chain-env"}


def test_fastapi_root(api_client) -> None:
    r = api_client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "env": "open-supply-chain-env"}


def test_fastapi_reset(api_client) -> None:
    r = api_client.post("/reset", json={"task": "task1"})
    assert r.status_code == 200
    body = r.json()
    assert body["cdc_inventory"] == 1500
    assert body["step"] == 0
    assert "pending_resupplies" in body
    assert "periodic_supply_rate" in body


def test_fastapi_step(api_client) -> None:
    api_client.post("/reset", json={"task": "task1"})
    r = api_client.post(
        "/step",
        json={"allocations": {"depotA": 100, "depotB": 100, "depotC": 100}},
    )
    assert r.status_code == 200
    body = r.json()
    assert "reward" in body
    assert "observation" in body


def test_fastapi_state(api_client) -> None:
    api_client.post("/reset", json={"task": "task1"})
    r = api_client.get("/state")
    assert r.status_code == 200
    body = r.json()
    assert "cdc_inventory" in body
    assert "periodic_supply_rate" in body
    assert "disruption_schedule" in body


def test_fastapi_step_invalid_400(api_client) -> None:
    api_client.post("/reset", json={"task": "task1"})
    r = api_client.post(
        "/step",
        json={"allocations": {"depotA": 99, "depotB": 0, "depotC": 0}},
    )
    assert r.status_code == 400


# =========================================================================
# 31–32. Inference stdout format tests
# =========================================================================

_STEP_RE = re.compile(
    r"^\[STEP\] step=\d+ "
    r"action=\{[^ ]+\} "
    r"reward=-?\d+\.\d{2} "
    r"done=(?:true|false) "
    r"error=.+$"
)


def test_inference_stdout_format(monkeypatch, capsys) -> None:
    """Patch _call_llm so no network call is made, then verify log format."""
    import inference

    monkeypatch.setattr(
        inference, "_call_llm",
        lambda obs, task: inference._build_fallback_action(obs),
    )

    inference.run_episode("static-baseline")
    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    # [START]
    assert lines[0].startswith("[START] ")
    assert "task=static-baseline" in lines[0]
    assert "env=open-supply-chain-env" in lines[0]
    assert "model=" in lines[0]

    # [STEP] lines
    step_lines = [l for l in lines if l.startswith("[STEP]")]
    assert len(step_lines) == 30
    for sl in step_lines:
        assert _STEP_RE.match(sl), f"Bad STEP format: {sl}"

    # [END]
    end_line = lines[-1]
    assert end_line.startswith("[END] ")
    assert "success=true" in end_line
    assert "steps=30" in end_line

    # rewards field: comma-separated, 2 decimal places, no spaces
    m = re.search(r"rewards=(.*)", end_line)
    assert m
    rewards_str = m.group(1)
    assert " " not in rewards_str
    parts = rewards_str.split(",")
    assert len(parts) == 30
    for p in parts:
        assert re.match(r"^-?\d+\.\d{2}$", p), f"Bad reward format: {p}"


def test_inference_end_printed_on_exception(monkeypatch, capsys) -> None:
    """[END] must appear even when env.step() raises an unhandled exception."""
    import inference

    monkeypatch.setattr(
        inference, "_call_llm",
        lambda obs, task: inference._build_fallback_action(obs),
    )

    call_count = 0
    original_step = SupplyChainEnv.step

    def exploding_step(self, action):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise RuntimeError("boom")
        return original_step(self, action)

    monkeypatch.setattr(SupplyChainEnv, "step", exploding_step)

    inference.run_episode("static-baseline")
    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    end_line = lines[-1]
    assert end_line.startswith("[END] ")
    assert "success=false" in end_line
