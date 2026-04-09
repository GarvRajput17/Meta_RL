# Open Supply Chain Environment

**Team QQuant** — Aaryan Antala, Nirbhay Sharma, Garv Grez (IIIT Bangalore)

---

## 1. Problem Statement

When a natural disaster strikes or supply hits scarcity when global order disrupts, authorities must redistribute scarce resources — food, water, fuel, medicine — from a central warehouse to multiple affected zones under rapidly changing conditions. This environment simulates that challenge as a **30-step sequential decision problem**: a single LLM agent acts as the central dispatcher, choosing how many units to ship from the Central Distribution Centre (CDC) to each depot while contending with road closures, demand spikes, and inventory cuts. All disruptions are deterministic and pre-defined per task, ensuring fully reproducible evaluation.

---

## 2. Network Topology

```
                    ┌── zone1  (demand: 60)
          ┌─ depotA─┤
          │   [600] └── zone2  (demand: 40)
          │
          │         ┌── zone3  (demand: 70)
  CDC ────┼─ depotB─┤
          │   [600] └── zone4  (demand: 30)
          │
          │         ┌── zone5  (demand: 55)
          └─ depotC─┤
              [600] └── zone6  (demand: 45)
```

- **CDC** → 3 depots (capacity 600 each) → 6 demand zones  
- **Edge naming**: `CDC->depotA`, `depotA->zone1`, etc.  
- Each edge can be `"open"` or `"closed"` at any step  
- Valid allocation values per depot: `{0, 50, 100, 200, 400}`

---

## 3. Task Definitions

### Task 1 — `static-baseline` (Easy)

A calm, no-disruption scenario to establish baseline performance. Supply significantly exceeds demand.

| Parameter | Value |
|---|---|
| **CDC Initial Stock** | 1500 |
| **Periodic Supply** | +250 / step |
| **Resupply Events** | None |
| **Disruptions** | None |
| **Demand Noise** | None |
| **Challenge** | Pure resource allocation — no surprises. The agent simply needs to distribute proportionally to zone demands each step. |

### Task 2 — `demand-spike` (Medium)

Demand spikes, a road closure, and stochastic noise force the agent to adapt in real time.

| Parameter | Value |
|---|---|
| **CDC Initial Stock** | 1400 |
| **Periodic Supply** | +170 / step |
| **Resupply Events** | +500 (step 10), +400 (step 20) |
| **Demand Noise** | σ = 5.0 (seed 42) — Gaussian noise added to base demands each step |
| **Disruption Timeline** | See below |

**Disruption schedule:**

| Step | Event |
|:---:|---|
| 4 |  zone3 demand spikes: 70 → **160** |
| 6 |  Road `depotA->zone2` **closes** |
| 10 |  Emergency resupply: **+500** to CDC |
| 12 |  zone3 demand normalises: 160 → **70** |
| 14 |  Road `depotA->zone2` **reopens** |
| 18 |  zone5 demand spikes: 55 → **120** |
| 20 |  Emergency resupply: **+400** to CDC |
| 24 |  zone5 demand normalises: 120 → **55** |

### Task 3 — `cascading-failure` (Hard)

A multi-phase crisis with cascading road closures, dual demand waves, two CDC inventory cuts, and a full depot blackout.

| Parameter | Value |
|---|---|
| **CDC Initial Stock** | 1200 |
| **Periodic Supply** | +200 / step |
| **Resupply Events** | +500 (step 15), +400 (step 22) |
| **Demand Noise** | None |
| **Disruption Timeline** | See below |

**Disruption schedule:**

| Step | Event |
|:---:|---|
| 2 |  Road `depotA->zone1` **closes** |
| 3 |  Road `CDC->depotB` **closes** (entire depot isolated) |
| 5 |  zone3 demand: 70 → **180**, zone6: 45 → **120** |
| 8 |  **CDC inventory cut to 50%** |
| 10 |  Road `depotA->zone1` **reopens** |
| 11 |  Road `CDC->depotB` **reopens** |
| 13 |  zone4 demand: 30 → **150**, zone3 normalises → **70** |
| 15 |  Emergency resupply: **+500** |
| 18 |  Road `CDC->depotC` **closes** (depot blackout) |
| 20 |  zone4: 150 → **30**, zone6: 120 → **45** |
| 22 |  Road `CDC->depotC` **reopens** + 📦 **+400** resupply |
| 25 |  **CDC inventory cut to 70%** |
| 27 |  zone1 demand: 60 → **130** (late-episode surge) |

---

## 4. Baseline Heuristic

The baseline is a **deterministic proportional-allocation heuristic** implemented in [`graders.py`](graders/graders.py). It uses no learning and no LLM — just simple math:

### Algorithm

```
1. For each depot d:
     if road CDC->d is CLOSED:
         target[d] = 0
     else:
         target[d] = Σ demand[z]  for all zones z served by d

2. total_target = Σ target[d]

3. If total_target ≤ CDC inventory:
     scaled[d] = target[d]              # send exactly what's needed
   Else:
     scaled[d] = target[d] × (CDC / total_target)  # scale down proportionally

4. For each depot d:
     alloc[d] = clamp_down(scaled[d])    # round down to nearest valid value
     alloc[d] = min(alloc[d], budget, headroom[d])
     budget  -= alloc[d]
```

`clamp_down(x)` rounds x **down** to the largest value in `{0, 50, 100, 200, 400}` that doesn't exceed x.

### Strengths & Limitations

| ✅ Strengths | ❌ Limitations |
|---|---|
| Zero latency — no API calls | No forward planning (doesn't look ahead at future supply/demand) |
| Deterministic & reproducible | No memory of past actions or reward feedback |
| Respects all hard constraints | Cannot anticipate or pre-position for disruptions |
| Good enough for static scenarios | Wastes supply during road closures / demand lulls |

---

## 5. Reward Function

Per-step reward is computed **after** allocation (CDC → depots) and distribution (depots → zones):

```
                         unmet demand across all zones
r  = − ─────────────────────────────────────────────────
                       total demand across all zones

r += 0.05 × (number of zones whose demand is exactly satisfied)

if last step (step 30) AND unmet demand == 0:
    r += 0.2                              # end-of-episode bonus
```

| Component | Range | Purpose |
|---|---|---|
| **Unmet demand penalty** | [−1, 0] | Core signal — fraction of total demand left unserved |
| **Exact satisfaction bonus** | [0, +0.30] | +0.05 for each of the 6 zones fully satisfied (encourages precision) |
| **Zero-unmet end bonus** | 0 or +0.20 | Terminal bonus for perfect final-step delivery |

**Invalid action penalty**: If the agent violates any constraint (over-allocates CDC, sends to closed road, exceeds depot headroom), the step returns `reward = −5.0`, **no state is mutated**, and the step counter does not advance.

### Normalised Score

Final score is computed from the **cumulative** reward over all 30 steps:

```
score = max(0, min(1, 1 + total_reward / 5))
```

This maps the total reward into `[0, 1]` where a total reward of 0 maps to 1.0 and −5 maps to 0.0.

> **Note**: In practice, scores are clamped to `(0.01, 0.99)` to satisfy the OpenEnv strict `(0, 1)` requirement.

---

## 6. How the RL Agent Uses Noise-Induced Updates

Unlike the static heuristic, our LLM agent operates as an **in-context reinforcement learner** that leverages stochastic demand noise (present in Task 2) and dynamic disruptions as learning signals:

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Agent (Llama 3.3 70B)                │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │ System   │   │ Observation  │   │ In-Context Memory   │  │
│  │ Prompt   │ + │ + Changes    │ + │ (reward & action    │  │
│  │ (rules)  │   │ + Recommend  │   │   history)          │  │
│  └──────────┘   └──────────────┘   └─────────────────────┘  │
│                          ↓                                   │
│               JSON action output                             │
│         {"allocations":{"depotA":X,...}}                      │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
            ┌─────────────────┐
            │  Action         │
            │  Sanitizer      │ ← Guarantees constraint compliance
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │  Environment    │ → reward, next observation
            └─────────────────┘
```

### The Feedback Loop

1. **Demand noise** (σ = 5.0 in Task 2) perturbs zone demands each step via Gaussian noise. This means the optimal allocation is never exactly the same twice — the agent must **read the actual demands** rather than memorise a static plan.

2. **Change detection**: At each step, the agent receives an explicit `== CHANGES THIS STEP ==` block highlighting what shifted since the last step (road status changes, demand deltas). This forces immediate reaction.

3. **Reward history as RL signal**: The complete reward trajectory is fed back into the prompt each step. The agent sees:
   - Whether its last action scored well or poorly
   - The cumulative average reward trend
   - Explicit guidance: *"YOUR ALLOCATION WAS BAD. Follow the recommendation below!"* when reward < 0

4. **Recommended allocation as policy prior**: The `_compute_recommended_allocation()` function computes a demand-proportional allocation accounting for:
   - **Future supply planning**: Budget considers `periodic_supply_rate × remaining_steps`
   - **Depot headroom**: Never exceeds physical capacity
   - **Road awareness**: Zero allocation to closed routes
   - **Leftover redistribution**: Surplus budget goes to highest-demand depots

5. **Action sanitizer as safety net**: Even if the LLM outputs invalid values, the sanitizer (`_sanitize_action`) ensures every submitted action is constraint-compliant — clamping to valid values, zeroing closed routes, and reducing over-budget allocations.

### Why This Outperforms the Heuristic

| Mechanism | Effect |
|---|---|
| **Multi-step memory** | Agent avoids repeating mistakes by seeing full reward history |
| **Change reactivity** | Explicit change detection triggers immediate adaptation |
| **Noise awareness** | Reads actual (noised) demands rather than assuming base values |
| **Forward planning** | Recommended allocation accounts for remaining supply pipeline |
| **LLM reasoning** | Can reason about trade-offs the heuristic's formula cannot express |

---

## 7. Performance Comparison

Results from running both the deterministic heuristic and the RL LLM agent across all three tasks:

| Task | Difficulty | Heuristic Score | RL LLM Agent Score | Δ Improvement |
|:---|:---:|:---:|:---:|:---:|
| `static-baseline` | 🟢 Easy | 0.99 | **0.99** | — |
| `demand-spike` | 🟡 Medium | 0.41 | **0.50** | +22% |
| `cascading-failure` | 🔴 Hard | 0.13 | **0.229** | +76% |

> **Score formula**: `max(0.01, min(0.99, 1 + total_reward / 5))`
> 
> Task 1 achieves near-perfect score for both agents (clamped to 0.99 by design).

### Key Takeaways

- **Task 1** (static): Both approaches score ~1.0. With zero disruptions and abundant supply, even a simple heuristic achieves near-optimal allocation. *No learning is needed.*

- **Task 2** (demand-spike): The RL agent gains **+22%** over the heuristic. The stochastic demand noise (σ = 5.0) and mid-episode disruptions create situations where the heuristic's one-step-lookahead fails — the LLM agent leverages its reward history to course-correct.

- **Task 3** (cascading-failure): The RL agent gains **+76%** over the heuristic. With cascading road closures, inventory cuts, and demand surges, the heuristic's greedy proportional strategy falls apart. The LLM agent's ability to reason about compound disruptions and adapt its allocation strategy step-by-step proves critical.

---

## 8. Setup & Running

### Environment Variables

| Variable | Required | Default |
|---|---|---|
| `API_BASE_URL` | No | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | No | `llama-3.3-70b-versatile` |
| `HF_TOKEN` | **Yes** | — |

### Local Run

```bash
pip install -r requirements.txt
export HF_TOKEN="your-token"
python inference.py             # runs all 3 tasks
python inference.py --task demand-spike  # single task
```

### Docker

```bash
docker build -t open-supply-chain-env .
docker run -e HF_TOKEN="your-token" -p 7860:7860 open-supply-chain-env
```

### Run Baseline Heuristic Grader

```bash
python -m graders.graders
```

### Tests

```bash
pytest tests/ -v
```

---

## 9. Inference Output Format

The script emits exactly three line types to stdout:

```
[START] task=static-baseline env=open-supply-chain-env model=llama-3.3-70b-versatile
[STEP] step=1 action={"allocations":{"depotA":100,"depotB":100,"depotC":100}} reward=0.30 done=false error=null
[STEP] step=2 action={"allocations":{"depotA":100,"depotB":50,"depotC":100}} reward=0.25 done=false error=null
...
[STEP] step=30 action={"allocations":{"depotA":0,"depotB":0,"depotC":0}} reward=-7.00 done=true error=null
[END] success=true steps=30 score=0.85 rewards=0.30,0.25,...,-7.00
```

---

## 10. OpenEnv Compliance

- ✅ Typed Pydantic v2 models: `Observation`, `Action`, `Reward`, `StepResult`
- ✅ `reset(task)` → initial `Observation`
- ✅ `step(action)` → `(Observation, reward, done, info)`
- ✅ `state()` → full internal state (deep copy)
- ✅ `openenv.yaml` with name, version, tasks, API endpoints
- ✅ 3 graded tasks (easy → medium → hard) with deterministic graders, scores in (0.0, 1.0)
- ✅ Root-level `inference.py` with exact `[START]`/`[STEP]`/`[END]` stdout format
- ✅ Working `Dockerfile` + Hugging Face Space deployment
- ✅ Docker: 2 vCPU / 8 GB RAM, inference < 20 minutes
