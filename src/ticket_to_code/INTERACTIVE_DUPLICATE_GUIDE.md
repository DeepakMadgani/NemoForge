# Interactive Duplicate Mode (Think Like Copilot)

This guide explains the new interactive layer that lets Ticket-to-Code behave like a transparent engineering copilot:

- Keep chat context per ticket session
- Show primary vs secondary hypotheses
- Show elimination rules
- Accept correction feedback
- Re-run with corrected reasoning

## Why this exists

Your goal is not only autonomous execution, but also **interactive correction**:

1. User gives ticket and context
2. System explains reasoning
3. User says what is wrong
4. System updates reasoning and re-runs

## Session Stages

The session manager now tracks these stages:

- `intake`
- `fact_extraction`
- `hypothesis`
- `ready_to_run`
- `running`
- `completed`
- `failed`

## New API Flow

### 1) Create conversation session

`POST /ticket-to-code/conversation/session`

Body:

```json
{
  "workspace_path": "C:/CC4E",
  "goal": "Fix registers upload button disabled after deliverables flow",
  "initial_message": "Ticket details and repro steps..."
}
```

Response includes:

- `session_id`
- current `stage`
- `reasoning` snapshot
- normalized conversation context

### 2) Add user messages

`POST /ticket-to-code/conversation/{session_id}/message`

Body:

```json
{
  "text": "Also ensure no DB migration files are touched"
}
```

This refreshes:

- Technical facts
- Primary hypothesis
- Secondary hypotheses
- Elimination rules
- Next actions

### 3) Read reasoning snapshot

`GET /ticket-to-code/conversation/{session_id}/reasoning`

This returns exactly what user needs to audit:

- stage
- confidence
- primary hypothesis
- secondary hypotheses
- elimination rules
- next actions
- open questions

### 4) Correct the system

`POST /ticket-to-code/conversation/{session_id}/feedback`

Body:

```json
{
  "feedback": "Primary hypothesis is wrong. issue is in shared form emitter state"
}
```

System stores rejected decision and re-enters extraction/hypothesis stage.

### 5) Run pipeline from conversation

`POST /ticket-to-code/conversation/{session_id}/run`

Body:

```json
{
  "technology": "nodejs",
  "max_fix_attempts": 3
}
```

The system synthesizes a ticket from chat context and executes the autonomous workflow.

## Reasoning Contract

Each session maintains:

- Primary hypothesis (best current root cause)
- Secondary hypotheses (alternatives)
- Elimination rules (what to avoid and why)
- Planned actions (what comes next)

This enables traceable, correction-friendly behavior.

## How this maps to your goal

For CC4E ticket solving, this gives:

1. Context continuity across messages
2. Explainability before code changes
3. Fast correction loop when reasoning is wrong
4. Clean bridge from chat to autonomous execution

## Recommended next phase

1. Persist sessions in DB/Redis (not in-memory)
2. Add per-session run history and diff links
3. Add confidence gate before execution (`run` blocked if confidence too low)
4. Add UI for stage timeline and evidence drill-down
