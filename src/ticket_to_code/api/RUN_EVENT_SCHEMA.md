# Run Event Schema (Backend-Owned)

This schema is the single contract between thin UI and backend orchestrator.
UI must not infer logic. UI only renders what backend emits.

## Transport

- Endpoint: `GET /ticket-to-code/runs/{run_id}/events`
- Protocol: Server-Sent Events (SSE)
- Content-Type: `text/event-stream`

## Event Envelope

Every non-heartbeat event uses this JSON envelope:

```json
{
  "run_id": "run_123abc",
  "seq": 1,
  "type": "stage.started",
  "stage": "classification",
  "status": "running",
  "timestamp": "2026-07-17T12:00:00Z",
  "payload": {}
}
```

Fields:
- `run_id`: backend run identifier
- `seq`: monotonic per run
- `type`: event type string
- `stage`: current backend stage
- `status`: backend run status
- `timestamp`: UTC ISO-8601
- `payload`: event-specific data

## Core Event Types

- `run.created`: run accepted and initialized
- `run.awaiting_approval`: paused for human approval
- `run.approved`: approval accepted and run resumed
- `run.failed_closed`: explicit fail-closed termination
- `stage.started`: stage has started
- `stage.completed`: stage has completed
- `run.need_more_info`: backend requires clarification
- `run.clarification_answered`: clarification answer accepted
- `run.retry`: backend retry attempt incremented
- `run.completed`: terminal completion event
- `run.failed`: runtime exception / terminal failure
- `heartbeat`: keep-alive event (no seq)

## Stage Names

Current standard stage names:
- `intake`
- `approval_gate`
- `classification`
- `strategy_selection`
- `discovery`
- `validation`
- `clarification`
- `completed`

Node-level live stages emitted from LangGraph stream:
- `node:<node_name>` (for example: `node:investigate`, `node:plan`, `node:generate_code`)

## REST Endpoints

1. Create run
- `POST /ticket-to-code/runs`
- Input: `ticket_title`, `description`, `constraints[]`, `repo_scope`, optional `acceptance_criteria[]`, `requires_approval`
- Output: `run_id`, `status`, `initial_status`

2. Stream run events
- `GET /ticket-to-code/runs/{run_id}/events`

3. Get run status
- `GET /ticket-to-code/runs/{run_id}/status`
- Output includes: `current_stage`, `retries_used`, `blocking_reason`

4. Get final report
- `GET /ticket-to-code/runs/{run_id}/final-report`
- Output includes: `findings`, `decisions`, `changed_files`, `validation_evidence`, `residual_risks`

5. Clarification
- `GET /ticket-to-code/runs/{run_id}/clarification`
- `POST /ticket-to-code/runs/{run_id}/clarification` with `{ "answer": "..." }`

6. Approval
- `POST /ticket-to-code/runs/{run_id}/approval` with `{ "approved": true|false, "note": "..." }`

## UI Rules

- UI must treat backend `status` and `type` as source of truth.
- UI should only render stage state, prompts, and final report.
- UI should not compute pass/fail from local heuristics.
