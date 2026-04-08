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

## 4. Tasks

| Task | Difficulty | Disruptions | Heuristic Score | RL Target |
|---|---|---|---|---|
| `static-baseline` | Easy | None — stationary demand (50/zone/step) | 0.00 | ≥ 0.50 |
| `demand-spike` | Medium | Step 5: zone3 demand → 150; Step 15: zone3 → 50 | 0.00 | ≥ 0.40 |
| `cascading-failure` | Hard | Step 3: CDC→depotB closed; Step 7: zone4 → 100; Step 11: road reopens; Step 12: CDC × 0.5 | 0.00 | ≥ 0.30 |

> **Note**: The proportional heuristic scores 0.00 because total supply (CDC 2000 + depot 600) cannot cover total demand (300 × 30 = 9000). An intelligent agent must triage — prioritising high-demand zones, pre-positioning stock, and avoiding depot stockouts — to outperform the heuristic.

---

## 5. Reward Function

Per-step reward computed **after** allocation and distribution:

```
r  = −(unmet_demand / total_demand)       # in [−1, 0]
r += 0.05 × (# zones exactly satisfied)   # up to +0.30
r −= 2.0  × (# depots with stockout)      # 0, −2, −4, or −6

if last_step and unmet == 0:
    r += 0.2                               # end-of-episode bonus
```

**Stockout condition**: a depot has 0 inventory after distribution AND its zones' total demand was not fully met.

**Invalid actions** (total exceeds CDC, allocation to a closed road, allocation exceeds depot headroom) receive `reward = −5.0`. No state is mutated and the step counter does not advance. There is **no soft road-violation penalty** — invalid road allocations are simply rejected.

**Normalised score**: `max(0, min(1, 1 + total_reward / 5))`

---

## 6. Setup

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

## 7. Inference Output Format

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

## 8. Grader Usage

Run the deterministic proportional-heuristic grader on all three tasks:

```bash
python -m graders.graders
```

Output is a JSON array with `task`, `total_reward`, `normalised_score`, `per_step_rewards`, and `policy` for each task.

---

## 9. Hugging Face Space

- The Space **must** be tagged `openenv`.
- The Space **must** be in the `Running` state before submission.
- Turn off all other Spaces — only keep the submission Space active to avoid build delays.
- The environment runs inside a Docker container limited to **2 vCPU / 8 GB RAM**. Ensure all dependencies fit within these constraints.
- Inference must complete within **20 minutes**.
