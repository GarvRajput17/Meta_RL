# Test Plan — OpenSupplyChainEnv

## 1. What Can Be Verified Without a Real API Key

Everything below has been tested and passes (38/38 pytest tests).

### 1.1 Environment Mechanics

| Test | Status |
|---|---|
| `reset()` returns valid Observation for all 3 tasks | Pass |
| Valid `step()` returns StepResult with reward in sane range | Pass |
| Invalid action (exceeds CDC) → reward = -5.0, error set | Pass |
| Invalid action does not mutate internal state | Pass |
| Invalid action does not advance the step counter | Pass |
| Pydantic rejects allocation value not in {0,50,100,200,400} | Pass |
| Pydantic rejects extra depot keys | Pass |
| Pydantic rejects missing depot keys | Pass |
| Full 30-step episode completes for all 3 tasks | Pass |
| `state()` returns a deep copy (mutation-safe) | Pass |
| Observation returned by `reset()` is safe to mutate externally | Pass |

### 1.2 Disruption Schedule

| Test | Status |
|---|---|
| task3 step 3: CDC→depotB closes | Pass |
| task2 step 5: zone3 demand → 150 | Pass |
| task2 step 15: zone3 demand → 50 | Pass |
| task3 step 11: CDC→depotB reopens | Pass |
| Same-step road_open wins over road_closure | Pass |
| Same-step multiple demand_change: last in schedule wins | Pass |
| Same-step multiple cdc_inventory_cut: factors compound | Pass |

### 1.3 Graders

| Test | Status |
|---|---|
| `run_heuristic_grader` returns normalised_score in [0, 1] for all tasks | Pass |
| `compute_normalised_score(0)` == 1.0 | Pass |
| `compute_normalised_score(-5)` == 0.0 | Pass |
| `compute_normalised_score(-2.5)` == 0.5 | Pass |
| Grader is deterministic across repeated calls | Pass |

### 1.4 Config & Deployment

| Test | Status |
|---|---|
| `openenv.yaml` contains expected tasks, api keys, entry_point | Pass |
| FastAPI `GET /health` returns 200 | Pass |
| FastAPI `GET /` returns health payload | Pass |
| FastAPI `POST /reset` returns Observation | Pass |
| FastAPI `POST /step` returns StepResult | Pass |
| FastAPI `GET /state` returns full state | Pass |
| FastAPI `POST /step` with invalid value returns HTTP 400 | Pass |

### 1.5 Inference Stdout Format

| Test | Status |
|---|---|
| `[START]` line has correct fields (task, env, model) | Pass |
| `[STEP]` lines match exact regex format | Pass |
| Action JSON is compact (no spaces) | Pass |
| Rewards have exactly 2 decimal places | Pass |
| `done` is lowercase `true`/`false` | Pass |
| `[END]` is always emitted, even on exception | Pass |
| `success=false` when exception occurs mid-episode | Pass |

---

## 2. What MUST Be Tested With a Real LLM (on HF Space or Locally)

These cannot be verified with mocks — they require actual LLM inference.

### 2.1 LLM Response Quality

| What to check | How | Expected |
|---|---|---|
| LLM returns valid JSON | Run `python inference.py --task static-baseline` | No fallback actions in output (all `error=null`) |
| LLM respects closed roads | Run task3, check steps 3–10 | `depotB` allocation should be 0 when road is closed |
| LLM adapts to demand spike | Run task2, check steps 5–14 | More units routed toward depotB (serves zone3) |
| LLM conserves CDC inventory | All tasks | Agent should not blindly drain CDC in early steps |
| LLM avoids depot stockout | All tasks | Fewer steps with reward ≤ -2.0 |

### 2.2 How to Run

```bash
export HF_TOKEN="<your-token>"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"

# Run each task
python inference.py --task static-baseline
python inference.py --task demand-spike
python inference.py --task cascading-failure

# Compare against heuristic
python -m graders.graders
```

### 2.3 What to Look For in the Output

1. **All `error=null`** — means the LLM produced valid actions every step.
2. **Rewards trending less negative** than the heuristic (-7.0 per step after CDC runs out).
3. **`[END] success=true`** — no unhandled exceptions.
4. **30 `[STEP]` lines** — full episode completed.

### 2.4 LLM vs Heuristic Comparison

After running both, compare:

| Metric | Heuristic | LLM Agent |
|---|---|---|
| Total reward (task1) | -150.95 | ? |
| Total reward (task2) | -154.07 | ? |
| Total reward (task3) | -149.07 | ? |
| Steps with zero unmet demand | ~8 | should be higher |
| Steps with depot stockout | ~22 | should be lower |

The LLM should outperform the heuristic by conserving CDC, triaging zones, and avoiding the -2.0 stockout penalty.

---

## 3. Pre-Submission Checklist

Before submitting to the hackathon:

- [ ] `pytest tests/ -v` — all 38 tests pass
- [ ] `python -m graders.graders` — runs without error, outputs JSON
- [ ] `python inference.py --task static-baseline` — completes with real LLM, `success=true`
- [ ] `python inference.py --task demand-spike` — completes with real LLM, `success=true`
- [ ] `python inference.py --task cascading-failure` — completes with real LLM, `success=true`
- [ ] `docker build -t open-supply-chain-env .` — builds successfully
- [ ] `docker run -e HF_TOKEN=... -p 7860:7860 open-supply-chain-env` — starts, `/health` returns 200
- [ ] HF Space deployed, tagged `openenv`, in `Running` state
- [ ] Inference completes within 20 minutes
- [ ] Container fits within 2 vCPU / 8 GB RAM



Can your env handle general resource scarcity, not just road closures?
Short answer: partially yes, but it's limited in scope right now. Here's an honest assessment:

What your env already models well:

Supply-demand mismatch (CDC 2600 total vs 9000 demand — genuine scarcity)
Demand shocks (zone3 suddenly tripling)
Infrastructure disruption (road closures cutting off entire depots)
Inventory shocks (CDC halved mid-episode)
Triage decisions under scarcity (which zones to prioritize when you can't serve everyone)
What it does NOT model yet (but could with the existing event system):

Multi-resource types — right now there's one generic "unit." Real disasters need food, water, medicine, fuel as separate resources with different priorities. Your env treats everything as fungible.

Demand heterogeneity — all zones start at 50. In a real crisis, a hospital zone might need 500 while a residential area needs 50. Your event system already supports demand_change, so you could set varied initial demands per task.

Supply replenishment — CDC inventory is fixed at episode start and only decreases. Real supply chains get resupply shipments. You'd need a new event type like cdc_resupply.

Depot-to-depot transfers — if depotB is overstocked and depotA is empty, there's no lateral transfer. The topology is strictly CDC→depot→zone.

Degraded capacity — roads are binary (open/closed). Real disruptions include partial capacity (bridge damaged, can only send 50% of normal throughput). You'd need a road_capacity field instead of just status.

Time-varying supply costs — in global scarcity, the "cost" of sending resources increases. No pricing mechanism exists.

The good news: your event system (disruption_schedule) is extensible. Adding new event types like cdc_resupply, depot_capacity_change, or road_partial_closure would follow the same pattern — a new Pydantic model, a new phase in _apply_disruptions, and new task JSONs. The core reset/step/state loop doesn't need to change.

For the hackathon submission: what you have is strong enough. The judges care about (1) real-world utility of the domain, (2) task quality, and (3) environment design. Your env models the core decision problem — scarce resources, disrupted infrastructure, multi-zone triage. The 3-task difficulty progression shows the LLM must reason adaptively. That's the story to tell.

For a real-world extension (post-hackathon): you'd want multiple resource types, stochastic demand, partial road degradation, and resupply events. That would make it genuinely useful for disaster response training.