# OpenEnv Round 1 Compliance Checklist

This checklist is distilled from:
- `Meta OpenEnv Hackathon_ Guidelines.md`
- `hackathon.pdf`
- `RL_Env_Proposal (3).pdf`

## Hard Requirements (Disqualification/Validation Gates)

- Root-level `inference.py` exists.
- `inference.py` uses OpenAI client (`from openai import OpenAI`) for LLM calls.
- `inference.py` reads:
  - `API_BASE_URL` with default
  - `MODEL_NAME` with default
  - `HF_TOKEN` mandatory
- `inference.py` emits exact stdout format:
  - one `[START] ...`
  - one `[STEP] ...` per env step
  - one `[END] ...` always (including errors)
  - booleans lowercase (`true`/`false`)
  - rewards fixed to 2 decimals
- OpenEnv interface exists and works:
  - typed Pydantic models (`Observation`, `Action`, reward type/model)
  - `reset(task)` returns initial observation
  - `step(action)` returns `(observation, reward, done, info)`
  - `state()` returns internal state
- `openenv.yaml` is valid and includes metadata + tasks.
- Minimum 3 tasks with deterministic graders and score range `[0.0, 1.0]`.
- Docker build works (`docker build` and container start).
- HF Space is deployed, tagged `openenv`, and running.
- Baseline inference run completes and reproduces scores.
- Runtime constraints respected:
  - < 20 min inference runtime
  - 2 vCPU / 8 GB RAM compatible

## Quality Requirements (Judging Weights)

- Real-world utility (30%)
- Task + grader quality (25%)
- Environment design (20%)
- Code/spec compliance (15%)
- Creativity/novelty (10%)

## Proposal-to-Implementation Mapping

- Domain: single-agent disaster supply chain allocation.
- Topology:
  - 1 CDC
  - 3 depots
  - 6 zones
  - road graph with disruptions
- Tasks:
  - easy: static baseline
  - medium: demand spike
  - hard: cascading failure
- Determinism:
  - disruption schedules in task JSON files
  - fixed seeds in graders
- Reward:
  - dense signal with partial progress + penalties
- Output:
  - no `score=` in `[END]` line (grader computes score separately)

## Build Order (Execution Plan)

1. Create package structure and base environment with Pydantic models.
2. Implement deterministic task JSON schemas and loader.
3. Implement environment simulation + reward function + done logic.
4. Implement 3 deterministic graders (`easy`, `medium`, `hard`) with normalized scoring.
5. Implement root `inference.py` with strict stdout protocol and error handling.
6. Add `openenv.yaml`, `README.md`, `requirements.txt`, `Dockerfile`.
7. Add local validation script:
   - OpenEnv compliance checks
   - stdout format sanity checks
   - baseline reproducibility check
8. Dry run locally under constrained settings and prepare HF Space deployment.

## Non-Negotiables for Coding Phase

- Keep APIs deterministic for evaluators.
- Avoid hidden randomness unless seeded and controlled.
- Keep inference/output parser-safe (single-line logs, strict field order).
- Prefer small dependencies to stay within 2 vCPU/8 GB limits.
- Ensure robust fallback behavior for malformed LLM responses.
