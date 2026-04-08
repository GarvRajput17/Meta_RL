"""
OpenSupplyChainEnv – Pydantic v2 models, topology, and environment logic.

Supply-chain topology
---------------------
Central Distribution Centre (CDC)
  ├── CDC->depotA ── depotA
  │                    ├── depotA->zone1 ── zone1
  │                    └── depotA->zone2 ── zone2
  ├── CDC->depotB ── depotB
  │                    ├── depotB->zone3 ── zone3
  │                    └── depotB->zone4 ── zone4
  └── CDC->depotC ── depotC
                       ├── depotC->zone5 ── zone5
                       └── depotC->zone6 ── zone6

Edge naming convention
----------------------
  CDC-to-depot edges : "CDC->depotA", "CDC->depotB", "CDC->depotC"
  Depot-to-zone edges: "depotA->zone1", "depotA->zone2",
                       "depotB->zone3", "depotB->zone4",
                       "depotC->zone5", "depotC->zone6"

All edges are directed.  Each can be "open" or "closed" at any step.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Tag, field_validator


# ---------------------------------------------------------------------------
# Topology constants
# ---------------------------------------------------------------------------

DEPOT_TO_ZONES: dict[str, list[str]] = {
    "depotA": ["zone1", "zone2"],
    "depotB": ["zone3", "zone4"],
    "depotC": ["zone5", "zone6"],
}

DEPOT_CAPACITY: dict[str, int] = {"depotA": 600, "depotB": 600, "depotC": 600}

DEPOTS: list[str] = ["depotA", "depotB", "depotC"]

ZONES: list[str] = ["zone1", "zone2", "zone3", "zone4", "zone5", "zone6"]

EDGES: tuple[str, ...] = (
    "CDC->depotA",
    "CDC->depotB",
    "CDC->depotC",
    "depotA->zone1",
    "depotA->zone2",
    "depotB->zone3",
    "depotB->zone4",
    "depotC->zone5",
    "depotC->zone6",
)

VALID_ALLOC: set[int] = {0, 50, 100, 200, 400}

MAX_STEPS: int = 30


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Full observable state handed to the LLM agent each step."""

    model_config = ConfigDict(strict=True, extra="forbid")

    cdc_inventory: int
    periodic_supply_rate: int
    depot_inventories: dict[str, int]
    zone_demands: dict[str, int]
    road_status: dict[str, str]
    step: int
    task_name: str
    pending_resupplies: list[dict[str, int]]


# ---------------------------------------------------------------------------
# Action (with strict depot-key and value validation)
# ---------------------------------------------------------------------------

class DepotAllocations(BaseModel):
    """
    Units to ship from CDC to each depot.

    Exactly three keys required: depotA, depotB, depotC.
    Each value must be in {0, 50, 100, 200, 400}.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    depotA: int
    depotB: int
    depotC: int

    @field_validator("depotA", "depotB", "depotC")
    @classmethod
    def _check_allowed(cls, v: int) -> int:
        if v not in VALID_ALLOC:
            raise ValueError(
                f"Allocation must be one of {sorted(VALID_ALLOC)}, got {v}"
            )
        return v


class Action(BaseModel):
    """Single action submitted by the agent per step."""

    model_config = ConfigDict(strict=True, extra="forbid")

    allocations: DepotAllocations


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Decomposed reward returned after each step."""

    model_config = ConfigDict(strict=True, extra="forbid")

    value: float
    unmet_demand_penalty: float
    exact_satisfaction_bonus: float
    zero_unmet_end_bonus: float


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Bundle returned by SupplyChainEnv.step()."""

    model_config = ConfigDict(strict=True, extra="forbid")

    observation: Observation
    reward: float
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Task-event models (deterministic disruption schedule)
# ---------------------------------------------------------------------------

class RoadClosureEvent(BaseModel):
    """Close an edge at *step*."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["road_closure"]
    step: int
    edge: str


class RoadOpenEvent(BaseModel):
    """Explicitly re-open an edge at *step*."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["road_open"]
    step: int
    edge: str


class DemandChangeEvent(BaseModel):
    """Set a zone's demand to *units* starting at *step*."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["demand_change"]
    step: int
    zone: str
    units: int


class InventoryCutEvent(BaseModel):
    """Multiply CDC inventory by *factor* at *step*."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["cdc_inventory_cut"]
    step: int
    factor: float


class CdcResupplyEvent(BaseModel):
    """Add *units* to CDC inventory at *step*."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["cdc_resupply"]
    step: int
    units: int


TaskEvent = Annotated[
    Union[
        Annotated[RoadClosureEvent, Tag("road_closure")],
        Annotated[RoadOpenEvent, Tag("road_open")],
        Annotated[DemandChangeEvent, Tag("demand_change")],
        Annotated[InventoryCutEvent, Tag("cdc_inventory_cut")],
        Annotated[CdcResupplyEvent, Tag("cdc_resupply")],
    ],
    Discriminator("type"),
]


# ---------------------------------------------------------------------------
# Task configuration (loaded from tasks/*.json)
# ---------------------------------------------------------------------------

class TaskConfig(BaseModel):
    """
    Complete specification for a single task.

    All disruptions are pre-defined in *disruption_schedule* so that
    evaluation is fully deterministic with no LLM calls.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    task_name: str
    cdc_initial_inventory: int
    periodic_supply_rate: int
    depot_initial_inventories: dict[str, int]
    base_zone_demands: dict[str, int]
    disruption_schedule: list[TaskEvent]


# ---------------------------------------------------------------------------
# Task-file validation helper
# ---------------------------------------------------------------------------

_REQUIRED_TOP_KEYS: set[str] = {
    "task_name",
    "cdc_initial_inventory",
    "periodic_supply_rate",
    "depot_initial_inventories",
    "base_zone_demands",
    "disruption_schedule",
}

_EVENT_REQUIRED_FIELDS: dict[str, set[str]] = {
    "road_closure": {"type", "step", "edge"},
    "road_open": {"type", "step", "edge"},
    "demand_change": {"type", "step", "zone", "units"},
    "cdc_inventory_cut": {"type", "step", "factor"},
    "cdc_resupply": {"type", "step", "units"},
}

_VALID_EDGES: set[str] = set(EDGES)


def validate_task_file(path: str) -> None:
    """Load a task JSON and raise ``ValueError`` on structural problems."""
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Task file must be a JSON object, got {type(data).__name__}")

    missing = _REQUIRED_TOP_KEYS - data.keys()
    if missing:
        raise ValueError(f"Missing top-level keys: {sorted(missing)}")

    extra = data.keys() - _REQUIRED_TOP_KEYS
    if extra:
        raise ValueError(f"Unknown top-level keys: {sorted(extra)}")

    if not isinstance(data["task_name"], str):
        raise ValueError("task_name must be a string")

    if not isinstance(data["cdc_initial_inventory"], int):
        raise ValueError("cdc_initial_inventory must be an int")

    if not isinstance(data["periodic_supply_rate"], int) or data["periodic_supply_rate"] < 0:
        raise ValueError("periodic_supply_rate must be a non-negative int")

    for key in ("depot_initial_inventories", "base_zone_demands"):
        if not isinstance(data[key], dict):
            raise ValueError(f"{key} must be a dict")

    depot_keys = set(data["depot_initial_inventories"].keys())
    if depot_keys != set(DEPOTS):
        raise ValueError(
            f"depot_initial_inventories must have keys {DEPOTS}, "
            f"got {sorted(depot_keys)}"
        )

    zone_keys = set(data["base_zone_demands"].keys())
    if zone_keys != set(ZONES):
        raise ValueError(
            f"base_zone_demands must have keys {ZONES}, "
            f"got {sorted(zone_keys)}"
        )

    if not isinstance(data["disruption_schedule"], list):
        raise ValueError("disruption_schedule must be a list")

    for i, event in enumerate(data["disruption_schedule"]):
        if not isinstance(event, dict):
            raise ValueError(f"disruption_schedule[{i}] must be a dict")

        etype = event.get("type")
        if etype not in _EVENT_REQUIRED_FIELDS:
            raise ValueError(
                f"disruption_schedule[{i}]: unknown event type {etype!r}"
            )

        required = _EVENT_REQUIRED_FIELDS[etype]
        emissing = required - event.keys()
        if emissing:
            raise ValueError(
                f"disruption_schedule[{i}] (type={etype}): "
                f"missing fields {sorted(emissing)}"
            )
        eextra = event.keys() - required
        if eextra:
            raise ValueError(
                f"disruption_schedule[{i}] (type={etype}): "
                f"unknown fields {sorted(eextra)}"
            )

        # Semantic: step must be in valid episode range
        step_val = event["step"]
        if not isinstance(step_val, int) or step_val < 0 or step_val > MAX_STEPS:
            raise ValueError(
                f"disruption_schedule[{i}] (type={etype}): "
                f"step must be an int in [0, {MAX_STEPS}], got {step_val!r}"
            )

        # Semantic: road edges must match the topology
        if etype in ("road_closure", "road_open"):
            if event["edge"] not in _VALID_EDGES:
                raise ValueError(
                    f"disruption_schedule[{i}] (type={etype}): "
                    f"edge {event['edge']!r} is not a valid edge; "
                    f"expected one of {sorted(_VALID_EDGES)}"
                )

        # Semantic: demand_change units must be non-negative
        if etype == "demand_change":
            if not isinstance(event["units"], int) or event["units"] < 0:
                raise ValueError(
                    f"disruption_schedule[{i}] (type={etype}): "
                    f"units must be a non-negative int, got {event['units']!r}"
                )
            if event["zone"] not in set(ZONES):
                raise ValueError(
                    f"disruption_schedule[{i}] (type={etype}): "
                    f"zone {event['zone']!r} is not a valid zone"
                )

        # Semantic: cdc_inventory_cut factor must satisfy 0 < factor <= 1
        if etype == "cdc_inventory_cut":
            fval = event["factor"]
            if not isinstance(fval, (int, float)) or fval <= 0 or fval > 1:
                raise ValueError(
                    f"disruption_schedule[{i}] (type={etype}): "
                    f"factor must satisfy 0 < factor <= 1, got {fval!r}"
                )

        # Semantic: cdc_resupply units must be a positive int
        if etype == "cdc_resupply":
            uval = event["units"]
            if not isinstance(uval, int) or uval <= 0:
                raise ValueError(
                    f"disruption_schedule[{i}] (type={etype}): "
                    f"units must be a positive int, got {uval!r}"
                )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupplyChainEnv:
    """Single-agent disaster resource allocation environment."""

    def __init__(self, tasks_dir: str = "tasks") -> None:
        self._tasks_dir = tasks_dir
        self._cdc: int | None = None
        self._periodic_rate: int = 0
        self._depots: dict[str, int] | None = None
        self._demands: dict[str, int] | None = None
        self._roads: dict[str, str] | None = None
        self._step: int | None = None
        self._task_name: str | None = None
        self._episode_rewards: list[float] | None = None
        self._schedule: list[dict[str, Any]] | None = None

    # ---- public API -------------------------------------------------------

    def reset(self, task: str) -> Observation:
        """Load a task JSON, initialise state, apply step-0 disruptions."""
        path = str(Path(self._tasks_dir) / f"{task}.json")
        validate_task_file(path)

        with open(path) as f:
            data = json.load(f)

        self._task_name = data["task_name"]
        self._cdc = data["cdc_initial_inventory"]
        self._periodic_rate = data["periodic_supply_rate"]
        self._depots = {k: v for k, v in data["depot_initial_inventories"].items()}
        self._demands = {k: v for k, v in data["base_zone_demands"].items()}
        self._roads = {e: "open" for e in EDGES}
        self._step = 0
        self._episode_rewards = []
        self._schedule = [dict(ev) for ev in data["disruption_schedule"]]

        self._apply_disruptions(0)
        return self._build_obs()

    def step(self, action: Action) -> StepResult:
        """Execute one environment step."""
        allocs = action.allocations

        # --- validation (return early, no state change) --------------------
        total_alloc = allocs.depotA + allocs.depotB + allocs.depotC
        if total_alloc > self._cdc:
            return self._invalid_result("total allocation exceeds CDC inventory")

        for depot in DEPOTS:
            units = getattr(allocs, depot)
            if self._roads[f"CDC->{depot}"] == "closed" and units != 0:
                return self._invalid_result(
                    f"road CDC->{depot} is closed but allocation is {units}"
                )
            headroom = DEPOT_CAPACITY[depot] - self._depots[depot]
            if units > headroom:
                return self._invalid_result(
                    f"{depot} headroom is {headroom} but allocation is {units}"
                )

        # --- allocation: CDC -> depots -------------------------------------
        for depot in DEPOTS:
            units = getattr(allocs, depot)
            if self._roads[f"CDC->{depot}"] == "open":
                self._depots[depot] += units
                self._cdc -= units

        # --- distribution: depots -> zones ---------------------------------
        units_received: dict[str, int] = {z: 0 for z in ZONES}
        for depot in DEPOTS:
            for zone in DEPOT_TO_ZONES[depot]:
                if self._roads[f"{depot}->{zone}"] == "open":
                    sent = min(self._depots[depot], self._demands[zone])
                    self._depots[depot] -= sent
                    units_received[zone] += sent

        # --- reward --------------------------------------------------------
        total_demand = sum(self._demands.values())
        assert total_demand > 0

        unmet = sum(
            max(0, self._demands[z] - units_received[z]) for z in self._demands
        )

        r = 0.0
        r -= unmet / total_demand
        r += 0.05 * sum(
            1 for z in ZONES if units_received[z] == self._demands[z]
        )

        if self._step == MAX_STEPS - 1 and unmet == 0:
            r += 0.2

        self._episode_rewards.append(r)

        # --- advance -------------------------------------------------------
        self._step += 1
        self._apply_disruptions(self._step)
        done = self._step >= MAX_STEPS

        return StepResult(
            observation=self._build_obs(),
            reward=r,
            done=done,
            info={"unmet_demand": unmet, "error": None},
        )

    def state(self) -> dict:
        """Return a deep copy of all internal state."""
        return copy.deepcopy(
            {
                "cdc_inventory": self._cdc,
                "periodic_supply_rate": self._periodic_rate,
                "depot_inventories": self._depots,
                "zone_demands": self._demands,
                "road_status": self._roads,
                "step": self._step,
                "task_name": self._task_name,
                "episode_rewards": self._episode_rewards,
                "disruption_schedule": self._schedule,
            }
        )

    def close(self) -> None:
        """No-op. Exists for API compatibility."""

    # ---- internal helpers -------------------------------------------------

    def _apply_disruptions(self, step: int) -> None:
        """Apply all events scheduled at *step* in fixed phase order."""
        events = [ev for ev in self._schedule if ev["step"] == step]

        known_types = {
            "road_closure", "road_open", "demand_change",
            "cdc_inventory_cut", "cdc_resupply",
        }
        for ev in events:
            if ev["type"] not in known_types:
                raise ValueError(f"Unknown disruption event type: {ev['type']!r}")

        # Phase 0: periodic supply pipeline (kicks in from step 1 onward)
        if step >= 1:
            self._cdc += self._periodic_rate

        # Phase 1: road closures
        for ev in events:
            if ev["type"] == "road_closure":
                self._roads[ev["edge"]] = "closed"

        # Phase 2: road opens (same-step open beats closure)
        for ev in events:
            if ev["type"] == "road_open":
                self._roads[ev["edge"]] = "open"

        # Phase 3: demand changes (last in schedule order wins per zone)
        for ev in events:
            if ev["type"] == "demand_change":
                self._demands[ev["zone"]] = ev["units"]

        # Phase 4: CDC inventory cuts (compound in schedule order)
        for ev in events:
            if ev["type"] == "cdc_inventory_cut":
                self._cdc = int(self._cdc * ev["factor"])

        # Phase 5: CDC resupply (additive, after cuts)
        for ev in events:
            if ev["type"] == "cdc_resupply":
                self._cdc += ev["units"]

    def _build_obs(self) -> Observation:
        """Construct an observation from current state (all copies)."""
        pending = [
            {"step": ev["step"], "units": ev["units"]}
            for ev in self._schedule
            if ev["type"] == "cdc_resupply" and ev["step"] > self._step
        ]
        return Observation(
            cdc_inventory=self._cdc,
            periodic_supply_rate=self._periodic_rate,
            depot_inventories=dict(self._depots),
            zone_demands=dict(self._demands),
            road_status=dict(self._roads),
            step=self._step,
            task_name=self._task_name,
            pending_resupplies=pending,
        )

    def _invalid_result(self, error_msg: str) -> StepResult:
        """Return a -5.0 penalty result without modifying any state."""
        return StepResult(
            observation=self._build_obs(),
            reward=-5.0,
            done=False,
            info={"error": error_msg},
        )
