"""Deterministic graders for the three supply-chain tasks."""

from __future__ import annotations

import json
import sys
from typing import Callable

from env.supply_chain_env import (
    Action,
    DEPOTS,
    DEPOT_TO_ZONES,
    DepotAllocations,
    MAX_STEPS,
    Observation,
    SupplyChainEnv,
    VALID_ALLOC,
)

SORTED_ALLOC = sorted(VALID_ALLOC)


def _clamp_to_valid(value: int) -> int:
    """Round *value* down to the largest element of VALID_ALLOC that does not exceed it."""
    result = 0
    for a in SORTED_ALLOC:
        if a <= value:
            result = a
        else:
            break
    return result


def proportional_heuristic_action(observation: Observation, env_state: dict) -> Action:
    """Deterministic proportional heuristic: allocate proportionally to zone demand."""
    cdc = observation.cdc_inventory
    demands = observation.zone_demands
    road_status = observation.road_status

    raw_targets: dict[str, int] = {}
    for depot in DEPOTS:
        if road_status[f"CDC->{depot}"] == "closed":
            raw_targets[depot] = 0
        else:
            raw_targets[depot] = sum(demands[z] for z in DEPOT_TO_ZONES[depot])

    total_target = sum(raw_targets.values())

    if total_target == 0 or cdc == 0:
        return Action(allocations=DepotAllocations(depotA=0, depotB=0, depotC=0))

    # Scale proportionally so total does not exceed CDC inventory
    scaled: dict[str, float] = {}
    if total_target <= cdc:
        scaled = {d: float(raw_targets[d]) for d in DEPOTS}
    else:
        for d in DEPOTS:
            scaled[d] = raw_targets[d] * (cdc / total_target)

    # Clamp each to nearest valid value not exceeding the scaled target,
    # then greedily ensure total does not exceed CDC
    allocs: dict[str, int] = {}
    budget = cdc
    for depot in DEPOTS:
        if raw_targets[depot] == 0:
            allocs[depot] = 0
            continue
        clamped = _clamp_to_valid(int(scaled[depot]))
        clamped = min(clamped, budget)
        clamped = _clamp_to_valid(clamped)
        allocs[depot] = clamped
        budget -= clamped

    return Action(allocations=DepotAllocations(**allocs))


def compute_normalised_score(total_reward: float) -> float:
    """Normalise cumulative reward to [0, 1]."""
    return round(max(0.0, min(1.0, 1 + total_reward / 5)), 4)


def run_grader(
    task: str,
    policy_fn: Callable[[Observation, dict], Action],
    tasks_dir: str = "tasks",
) -> dict:
    """Run a full episode with *policy_fn* and return grading results."""
    env = SupplyChainEnv(tasks_dir)
    obs = env.reset(task)

    per_step_rewards: list[float] = []
    done = False

    for _ in range(MAX_STEPS):
        if done:
            break
        action = policy_fn(obs, env.state())
        result = env.step(action)
        obs = result.observation
        per_step_rewards.append(result.reward)
        done = result.done

    env.close()

    total_reward = sum(per_step_rewards)
    return {
        "task": task,
        "total_reward": total_reward,
        "normalised_score": compute_normalised_score(total_reward),
        "per_step_rewards": per_step_rewards,
        "policy": policy_fn.__name__ if hasattr(policy_fn, "__name__") else str(policy_fn),
    }


def run_heuristic_grader(task: str) -> dict:
    """Grade the proportional heuristic on a single task."""
    result = run_grader(task, proportional_heuristic_action)
    result["policy"] = "proportional_heuristic"
    return result


if __name__ == "__main__":
    tasks = ["task1", "task2", "task3"]
    results = []
    for t in tasks:
        r = run_heuristic_grader(t)
        results.append(r)
    print(json.dumps(results, indent=2))
    sys.exit(0)
