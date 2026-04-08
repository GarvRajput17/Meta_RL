---
title: Open Supply Chain Env
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# OpenSupplyChainEnv

Single-agent LLM-driven disaster resource allocation environment for the [Meta OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv).

**Team QQuant** — Aaryan Antala, Nirbhay Sharma, Garv Grez (IIIT Bangalore)

---

## 1. Project Overview

When a natural disaster strikes, authorities must redistribute scarce resources — food, water, fuel, medicine — from a central warehouse to multiple affected zones under rapidly changing conditions. This environment simulates that challenge as a 30-step sequential decision problem: a single LLM agent acts as the central dispatcher, choosing how many units to ship from the Central Distribution Centre (CDC) to each depot while contending with road closures, demand spikes, and inventory cuts. All disruptions are deterministic and pre-defined per task, ensuring fully reproducible evaluation.

---

## 2. Topology

```
                    ┌── zone1
          ┌─ depotA─┤
          │         └── zone2
          │
          │         ┌── zone3
  CDC ────┼─ depotB─┤
          │         └── zone4
          │
          │         ┌── zone5
          └─ depotC─┤
                    └── zone6
```

**Edge naming**: `CDC->depotA`, `depotA->zone1`, etc.  
Each edge can be `"open"` or `"closed"` at any step.

---

## 3. OpenEnv Compliance Checklist

- ✅ Typed Pydantic v2 models: `Observation`, `Action`, `Reward`, `StepResult`
- ✅ `reset(task)` → initial `Observation`
- ✅ `step(action)` → `(Observation, reward, done, info)`
- ✅ `state()` → full internal state (deep copy)
- ✅ `openenv.yaml` with name, version, tasks, API endpoints
- ✅ 3 graded tasks (easy → medium → hard) with deterministic graders, scores in [0.0, 1.0]
- ✅ Root-level `inference.py` with exact `[START]`/`[STEP]`/`[END]` stdout format
- ✅ Working `Dockerfile` + Hugging Face Space deployment

---

## 4. Supply Economy

The CDC is fed by three supply streams:

| Source | Description |
|---|---|
| **Base stock** | `cdc_initial_inventory` — available from step 0 |
| **Procurement pipeline** | `periodic_supply_rate` — units added to CDC every step (from step 1) |
| **One-time deliveries** | `cdc_resupply` events — emergency reserves, refugee aid, international relief |

The economy is balanced so that total supply roughly matches total demand (~1.0–1.05×), meaning a well-played episode can achieve near-zero unmet demand.

## 5. Tasks

| Task | Difficulty | Base Stock | Periodic | Resupply Events | Disruptions |
|---|---|---|---|---|---|
| `static-baseline` | Easy | 1500 | 250/step | — | None |
| `demand-spike` | Medium | 1400 | 170/step | +500 (step 10), +400 (step 20) | zone3 spike (4-12), depotA->zone2 closure (6-14), zone5 spike (18-24) |
| `cascading-failure` | Hard | 1200 | 200/step | +500 (step 15), +400 (step 22) | Multi-phase: zone/CDC road closures, dual demand waves, two CDC cuts, depotC blackout |

The observation includes `pending_resupplies` so the agent can plan around upcoming deliveries.

---

## 6. Reward Function

Per-step reward computed **after** allocation and distribution:

```
r  = −(unmet_demand / total_demand)       # in [−1, 0]
r += 0.05 × (# zones exactly satisfied)   # up to +0.30

if last_step and unmet == 0:
    r += 0.2                               # end-of-episode bonus
```

The reward is driven by **unmet demand** (the fraction of total demand not served) and a small bonus for zones whose demand is exactly met. There is no depot stockout penalty — if a depot runs out of stock, the shortfall is already captured by the unmet demand term.

**Invalid actions** (total exceeds CDC, allocation to a closed road, allocation exceeds depot headroom) receive `reward = −5.0`. No state is mutated and the step counter does not advance. There is **no soft road-violation penalty** — invalid road allocations are simply rejected.

**Normalised score**: `max(0, min(1, 1 + total_reward / 5))`

---

## 7. Setup

### Environment Variables

| Variable | Required | Default |
|---|---|---|
| `API_BASE_URL` | No | `https://api.openai.com/v1` |
| `MODEL_NAME` | No | `gpt-4o-mini` |
| `HF_TOKEN` | **Yes** | — |

### Local Run

```bash
pip install -r requirements.txt
export HF_TOKEN="your-token"
python inference.py --task static-baseline
```

### Docker

```bash
docker build -t open-supply-chain-env .
docker run -e HF_TOKEN="your-token" -p 7860:7860 open-supply-chain-env
```

### Tests

```bash
pytest tests/ -v
```

---

## 8. Inference Output Format

The script emits exactly three line types to stdout:

```
[START] task=static-baseline env=open-supply-chain-env model=gpt-4o-mini
[STEP] step=1 action={"allocations":{"depotA":100,"depotB":100,"depotC":100}} reward=0.30 done=false error=null
[STEP] step=2 action={"allocations":{"depotA":100,"depotB":50,"depotC":100}} reward=0.25 done=false error=null
...
[STEP] step=30 action={"allocations":{"depotA":0,"depotB":0,"depotC":0}} reward=-7.00 done=true error=null
[END] success=true steps=30 rewards=0.30,0.25,...,-7.00
```

- One `[START]` at episode begin.
- One `[STEP]` per step, immediately after `env.step()` returns.
- One `[END]` after `env.close()`, **always** emitted (even on exception).
- `reward` and `rewards` formatted to 2 decimal places.
- `done` and `success` are lowercase: `true` or `false`.
- `error` is the raw error string or the literal word `null`.

---

## 9. Grader Usage

Run the deterministic proportional-heuristic grader on all three tasks:

```bash
python -m graders.graders
```

Output is a JSON array with `task`, `total_reward`, `normalised_score`, `per_step_rewards`, and `policy` for each task.

---

## 10. Hugging Face Space

- The Space **must** be tagged `openenv`.
- The Space **must** be in the `Running` state before submission.
- Turn off all other Spaces — only keep the submission Space active to avoid build delays.
- The environment runs inside a Docker container limited to **2 vCPU / 8 GB RAM**. Ensure all dependencies fit within these constraints.
- Inference must complete within **20 minutes**.
