"""
Root-level inference script for OpenSupplyChainEnv.

Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
Uses the OpenAI Python client for all LLM calls.
Emits exact [START]/[STEP]/[END] stdout format required by OpenEnv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI

from env.supply_chain_env import (
    Action,
    DEPOT_CAPACITY,
    DEPOTS,
    DEPOT_TO_ZONES,
    DepotAllocations,
    MAX_STEPS,
    Observation,
    SupplyChainEnv,
    VALID_ALLOC,
)
from graders.graders import compute_normalised_score


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Task-file mapping
# ---------------------------------------------------------------------------

TASK_FILE_MAP: dict[str, str] = {
    "static-baseline": "task1",
    "demand-spike": "task2",
    "cascading-failure": "task3",
}

SORTED_ALLOC = sorted(VALID_ALLOC)

# ---------------------------------------------------------------------------
# System prompt (exact)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a disaster logistics coordinator optimising supply allocation.\n"
    "Rules:\n"
    "- Allocate supplies from CDC to 3 depots each step. Depots auto-distribute to zones.\n"
    "- Goal: MINIMISE unmet demand. Reward = -unmet/total_demand + 0.05 per fully-satisfied zone.\n"
    "- CRITICAL: Match allocations to depot demand. If depotB has 100 demand and depotA has 100, they need similar allocations.\n"
    "- If a road is closed, that depot MUST get 0.\n"
    "- React IMMEDIATELY to demand changes — increase allocation to affected depots the same step.\n"
    "- React IMMEDIATELY to road closures/openings — redistribute to open depots.\n"
    "- A recommended allocation is provided each step. Follow it unless you have a strong reason to deviate.\n"
    "- Output ONLY valid JSON. No explanation, no markdown, no extra text."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_down(value: int) -> int:
    """Largest element of VALID_ALLOC that does not exceed *value*."""
    result = 0
    for a in SORTED_ALLOC:
        if a <= value:
            result = a
        else:
            break
    return result


def _clamp_up(value: int) -> int:
    """Smallest element of VALID_ALLOC that is >= *value*, or largest."""
    for a in SORTED_ALLOC:
        if a >= value:
            return a
    return SORTED_ALLOC[-1]


def _compute_recommended_allocation(observation: Observation) -> dict[str, int]:
    """Compute demand-proportional allocation as a recommendation to the LLM."""
    cdc = observation.cdc_inventory
    demands = observation.zone_demands
    road_status = observation.road_status
    depot_inv = observation.depot_inventories
    remaining_steps = MAX_STEPS - observation.step

    # Compute per-depot demand (only for open roads)
    depot_demand: dict[str, int] = {}
    for depot in DEPOTS:
        if road_status[f"CDC->{depot}"] == "closed":
            depot_demand[depot] = 0
        else:
            depot_demand[depot] = sum(
                demands[z] for z in DEPOT_TO_ZONES[depot]
                if road_status.get(f"{depot}->{z}") == "open"
            )

    total_demand = sum(depot_demand.values())
    if total_demand == 0 or cdc == 0:
        return {"depotA": 0, "depotB": 0, "depotC": 0}

    # Budget: account for periodic supply over remaining steps
    total_supply = cdc + observation.periodic_supply_rate * max(remaining_steps - 1, 0)
    total_future_demand = total_demand * remaining_steps
    # Allocate proportionally but don't overshoot what's needed
    budget_this_step = min(cdc, max(total_demand, total_supply // max(remaining_steps, 1)))

    allocs: dict[str, int] = {}
    budget = budget_this_step
    for depot in DEPOTS:
        if depot_demand[depot] == 0:
            allocs[depot] = 0
            continue
        # Proportional share
        share = depot_demand[depot] / total_demand * budget_this_step
        headroom = DEPOT_CAPACITY[depot] - depot_inv.get(depot, 0)
        # Clamp up to nearest valid to better meet demand
        clamped = _clamp_down(min(int(share + 0.5), budget, headroom))
        allocs[depot] = clamped
        budget -= clamped

    # If we have leftover budget and depots with unmet demand, distribute more
    if budget >= SORTED_ALLOC[1]:
        for depot in sorted(DEPOTS, key=lambda d: depot_demand[d], reverse=True):
            if depot_demand[depot] == 0:
                continue
            headroom = DEPOT_CAPACITY[depot] - depot_inv.get(depot, 0) - allocs[depot]
            extra = _clamp_down(min(budget, headroom))
            if extra > 0:
                allocs[depot] += extra
                budget -= extra

    return allocs


def _build_fallback_action(observation: Observation) -> Action:
    """Deterministic safe fallback that never violates any constraint."""
    cdc = observation.cdc_inventory
    road_status = observation.road_status
    depot_inv = observation.depot_inventories

    open_depots = [
        d for d in DEPOTS if road_status[f"CDC->{d}"] == "open"
    ]

    if not open_depots or cdc == 0:
        return Action(allocations=DepotAllocations(depotA=0, depotB=0, depotC=0))

    per_depot = cdc // len(open_depots)
    allocs: dict[str, int] = {}
    budget = cdc
    for depot in DEPOTS:
        if depot not in open_depots:
            allocs[depot] = 0
            continue
        headroom = DEPOT_CAPACITY[depot] - depot_inv.get(depot, 0)
        clamped = _clamp_down(min(per_depot, budget, headroom))
        allocs[depot] = clamped
        budget -= clamped

    return Action(allocations=DepotAllocations(**allocs))


def _build_user_message(
    observation: Observation,
    task_name: str,
    reward_history: list[float] | None = None,
) -> str:
    """Construct the user prompt including observation, constraints, and recommendation."""
    obs = observation
    depot_inv = obs.depot_inventories
    demands = obs.zone_demands

    closed_depots = [
        d for d in DEPOTS if obs.road_status.get(f"CDC->{d}") == "closed"
    ]
    closed_zone_roads = [
        edge for edge, st in obs.road_status.items()
        if st == "closed" and not edge.startswith("CDC->")
    ]

    # Per-depot demand summary
    depot_summaries: list[str] = []
    for depot in DEPOTS:
        zones = DEPOT_TO_ZONES[depot]
        zone_detail = ", ".join(f"{z}={demands[z]}" for z in zones)
        total_d = sum(demands[z] for z in zones)
        headroom = DEPOT_CAPACITY[depot] - depot_inv[depot]
        status = "CLOSED" if depot in closed_depots else "open"
        depot_summaries.append(
            f"  - {depot} (road: {status}, inv: {depot_inv[depot]}, "
            f"headroom: {headroom}, demand: {zone_detail}, total: {total_d})"
        )

    remaining_steps = MAX_STEPS - obs.step

    pending = obs.pending_resupplies
    if pending:
        resupply_lines = ", ".join(
            f"step {r['step']}: +{r['units']}" for r in pending
        )
    else:
        resupply_lines = "none"

    # Compute recommended allocation
    rec = _compute_recommended_allocation(obs)

    msg = f"STEP {obs.step} / 30  |  Task: {task_name}\n\n"

    # Show recent reward feedback if available
    if reward_history:
        recent = reward_history[-3:]
        trend = ", ".join(f"{r:.2f}" for r in recent)
        msg += f"== RECENT REWARDS == {trend}\n"
        if any(r < 0 for r in recent):
            msg += "WARNING: Negative rewards indicate unmet demand. Increase allocations!\n"
        msg += "\n"

    msg += (
        f"== SUPPLY ==\n"
        f"- CDC inventory: {obs.cdc_inventory}\n"
        f"- Periodic pipeline: +{obs.periodic_supply_rate}/step\n"
        f"- Upcoming deliveries: {resupply_lines}\n"
        f"- Remaining steps: {remaining_steps}\n\n"

        f"== DEPOTS ==\n"
        + "\n".join(depot_summaries) + "\n\n"

        f"== CONSTRAINTS ==\n"
        f"- Allocations must be in {sorted(VALID_ALLOC)}\n"
        f"- Sum ≤ {obs.cdc_inventory}\n"
        f"- Closed roads MUST get 0: {closed_depots if closed_depots else 'none'}\n"
        f"- Closed zone roads: {closed_zone_roads if closed_zone_roads else 'none'}\n"
        f"- Each allocation ≤ depot headroom\n\n"

        f'== RECOMMENDED == {{"allocations":{{"depotA":{rec["depotA"]},"depotB":{rec["depotB"]},"depotC":{rec["depotC"]}}}}}\n'
        f"Use this unless you have a specific reason to deviate.\n\n"

        f"== RESPONSE (JSON only) ==\n"
        f'{{"allocations":{{"depotA":<int>,"depotB":<int>,"depotC":<int>}}}}'
    )
    return msg


def _sanitize_action(observation: Observation, raw_allocs: dict[str, int]) -> Action:
    """Clamp LLM-proposed allocations so they never violate any constraint."""
    cdc = observation.cdc_inventory
    road_status = observation.road_status
    depot_inv = observation.depot_inventories

    allocs: dict[str, int] = {}
    for depot in DEPOTS:
        proposed = max(0, raw_allocs.get(depot, 0))
        if road_status.get(f"CDC->{depot}") == "closed":
            allocs[depot] = 0
            continue
        headroom = DEPOT_CAPACITY[depot] - depot_inv.get(depot, 0)
        allocs[depot] = _clamp_down(min(proposed, headroom))

    total = sum(allocs.values())
    if total > cdc:
        for depot in sorted(DEPOTS, key=lambda d: allocs[d], reverse=True):
            while allocs[depot] > 0 and sum(allocs.values()) > cdc:
                current = allocs[depot]
                lower = [v for v in SORTED_ALLOC if v < current]
                allocs[depot] = lower[-1] if lower else 0

    return Action(allocations=DepotAllocations(**allocs))


def _call_llm(
    observation: Observation,
    task_name: str,
    reward_history: list[float] | None = None,
) -> Action:
    """Call the LLM and parse its response into an Action, with fallback."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_message(observation, task_name, reward_history),
                },
            ],
            max_tokens=128,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines).strip()

        data = json.loads(raw)
        raw_allocs = {
            "depotA": int(data["allocations"]["depotA"]),
            "depotB": int(data["allocations"]["depotB"]),
            "depotC": int(data["allocations"]["depotC"]),
        }
        return _sanitize_action(observation, raw_allocs)
    except Exception:
        # Fallback: use the recommendation directly
        rec = _compute_recommended_allocation(observation)
        return _sanitize_action(observation, rec)


def _compact_action(action: Action) -> str:
    """Compact JSON representation of an action for stdout logging."""
    return json.dumps(action.model_dump(), separators=(",", ":"))


def _format_error(info: dict[str, Any]) -> str:
    """Format the error field for [STEP] output."""
    err = info.get("error")
    if err is None:
        return "null"
    return str(err)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(task: str) -> None:
    """Run a single episode and emit [START]/[STEP]/[END] to stdout."""
    task_file = TASK_FILE_MAP[task]
    env = SupplyChainEnv(tasks_dir="tasks")
    rewards: list[float] = []
    success = True
    steps_taken = 0

    try:
        obs = env.reset(task_file)
        print(f"[START] task={task} env=open-supply-chain-env model={MODEL_NAME}")

        done = False
        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            action = _call_llm(obs, task, rewards if rewards else None)
            result = env.step(action)

            obs = result.observation
            rewards.append(result.reward)
            done = result.done
            steps_taken = step_num

            action_str = _compact_action(action)
            error_str = _format_error(result.info)

            print(
                f"[STEP] step={step_num} "
                f"action={action_str} "
                f"reward={result.reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={error_str}"
            )

    except Exception:
        success = False
    finally:
        env.close()
        total_reward = sum(rewards)
        score = compute_normalised_score(total_reward)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={steps_taken} "
            f"rewards={rewards_str}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSupplyChainEnv inference")
    parser.add_argument(
        "--task",
        choices=["static-baseline", "demand-spike", "cascading-failure"],
        default="static-baseline",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Accepted silently and ignored.",
    )
    args = parser.parse_args()
    run_episode(args.task)
