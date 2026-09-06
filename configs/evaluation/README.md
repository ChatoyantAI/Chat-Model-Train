# Evaluation configuration

All paths in the example environment files are relative to the repository root.
Copy only the required files to their `.env` counterparts, then adjust paths and
set server-only keys. Actual `.env` files are ignored by Git; use mode `0600` on
POSIX hosts. Never put credentials in CLI arguments or browser code.

| Template | Service |
|---|---|
| `annotation.env.example` | Human annotation, replay, Judge, comparison and Battle |
| `filter-v2.env.example` | Isolated pointwise probability scoring |
| `conversation.env.example` | Conversation filtering, ranking and exports |

Copy provider settings as needed from
[`llm-judge.env.example`](../../src/evaluation/roleplay_annotation_platform/llm-judge.env.example).
The queue requires **one Uvicorn worker per service**. The conversation service
has its own 100-slot pool; other Judge tasks share the configurable 64-slot pool.
Neither browsing nor request preview should start paid requests.

The versioned systemd user-service templates assume the repository is checked out
at `%h/roleplay-evaluation`, with its virtual environment at `.venv` and private
environment files in `configs/evaluation/`. Edit those paths for your installation.
Use an authenticated HTTPS reverse proxy or a trusted private network. A username
is a result namespace, not password authentication.

No supervisor or recurring job is enabled by installation. To recover an existing
conversation task, run `supervise_conversation_run` explicitly with the task ID,
data and result paths, output root, and `--authorize-failure-retries`. It respects
the task's own dispatch budget and preserves manual pauses and cancellations.
