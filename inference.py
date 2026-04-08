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

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
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
    "You are a disaster logistics coordinator. Your job:\n"
    "- Allocate supplies from a Central Distribution Centre (CDC) to 3 depots.\n"
    "- Each depot automatically distributes to its zones to meet demand.\n"
    "- Your goal: minimise unmet demand across all zones every step.\n"
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


def _build_user_message(observation: Observation, task_name: str) -> str:
    """Construct the user prompt including observation and constraints."""
    obs = observation
    depot_inv = obs.depot_inventories
    demands = obs.zone_demands

    closed_depots = [
        d for d in DEPOTS if obs.road_status.get(f"CDC->{d}") == "closed"
    ]
    open_depots = [d for d in DEPOTS if d not in closed_depots]

    # Per-depot demand summary
    depot_summaries: list[str] = []
    for depot in DEPOTS:
        zones = DEPOT_TO_ZONES[depot]
        zone_detail = ", ".join(f"{z}={demands[z]}" for z in zones)
        total_d = sum(demands[z] for z in zones)
        headroom = DEPOT_CAPACITY[depot] - depot_inv[depot]
        status = "CLOSED" if depot in closed_depots else "open"
        depot_summaries.append(
            f"  - {depot} (road: {status}, inventory: {depot_inv[depot]}, "
            f"headroom: {headroom}, zone demand: {zone_detail}, total: {total_d})"
        )

    remaining_steps = MAX_STEPS - obs.step
    budget_hint = obs.cdc_inventory // remaining_steps if remaining_steps > 0 else 0

    msg = (
        f"STEP {obs.step} / 30  |  Task: {task_name}\n\n"

        f"== SUPPLY ==\n"
        f"- CDC inventory: {obs.cdc_inventory}\n"
        f"- Remaining steps: {remaining_steps}\n"
        f"- Suggested total budget this step: ~{budget_hint}\n\n"

        f"== DEPOTS ==\n"
        + "\n".join(depot_summaries) + "\n\n"

        f"== CONSTRAINTS (violating any → instant -5 penalty, step wasted) ==\n"
        f"- Each depot allocation must be one of: {sorted(VALID_ALLOC)}\n"
        f"- Sum of all allocations must be ≤ {obs.cdc_inventory} (CDC inventory)\n"
        f"- Closed-road depots MUST get 0: {closed_depots if closed_depots else 'none currently'}\n"
        f"- Each depot allocation must fit within its headroom (capacity {list(DEPOT_CAPACITY.values())[0]} - current inventory)\n\n"

        f"== STRATEGY TIPS ==\n"
        f"- Depots auto-distribute to their zones. You only control CDC→depot.\n"
        f"- Prioritise depots whose zones have highest total demand.\n"
        f"- Spread CDC across all 30 steps — don't spend it all early.\n\n"
    )

    # Few-shot example for non-easy tasks
    if task_name in ("demand-spike", "cascading-failure"):
        msg += (
            "== EXAMPLE (when CDC->depotB is closed) ==\n"
            '{"allocations":{"depotA":200,"depotB":0,"depotC":200}}\n\n'
        )

    msg += (
        "== YOUR RESPONSE ==\n"
        "Output ONLY this JSON (no text around it):\n"
        '{"allocations":{"depotA":<int>,"depotB":<int>,"depotC":<int>}}'
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


def _call_llm(observation: Observation, task_name: str) -> Action:
    """Call the LLM and parse its response into an Action, with fallback."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(observation, task_name)},
            ],
            max_tokens=256,
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
        return _build_fallback_action(observation)


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

            action = _call_llm(obs, task)
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
