# Lusy turn-level semantic audit v1

This append-only stage builds multilingual dense embeddings for semantic-repeat
auditing. It does not modify the frozen v2.2 JSONL, does not overwrite
`01_independent_v1`, and does not delete samples.

## Frozen contract

- Input dialogue pool: exact-dedup survivors from `01_independent_v1`.
- Exact scale: 1,254,267 dialogues and 18,958,570 retained turn pairs.
- One turn row contains the preceding user message, assistant response, stable
  IDs, language, model family, and per-assistant-event evaluation.
- Model: `BAAI/bge-m3`, revision
  `5617a9f61b028005a4858fdac845db406aefb181`.
- Dense representation: normalized CLS vector, 1,024 dimensions, float16.
- Maximum encoded length: 512 tokens for this v1 throughput/quality tradeoff.
- Both `assistant_text` and `user_text` are encoded. This enables the rule
  "assistant stayed semantically similar while the user changed".

Raw vectors require about 36.2 GiB per field, or 72.3 GiB for both fields,
before FAISS indexes and metadata.

## Remote artifact

```text
02_data_juicer/02_semantic_audit_v1/
  01_turn_view/
    bucket-000.parquet ... bucket-255.parquet
    reports/
  assistant_embeddings/
    bucket-000.f16.npy ... bucket-255.f16.npy
    manifests/
  user_embeddings/
    bucket-000.f16.npy ... bucket-255.f16.npy
    manifests/
  07_dashboard/
    run_status.json
    embedding_benchmark.json
```

Each Parquet/vector shard is written atomically. A completed manifest makes the
embedding stage resumable without recomputing finished buckets.

## Execution

```bash
./install_runtime.sh
./run_full.sh
```

The deployed user service is `lusy-data-juicer-semantic-v1.service`. The shared
Streamlit UI exposes `/Lusy_Semantic_Audit` on port 8988.

## Deliberately deferred

This version only builds the reusable turn view and vectors. Cosine candidate
generation, within-dialog windows, FAISS indexes, semantic thresholds,
clustering, and quality-aware representative selection belong to the next
calibrated stage. No fixed cosine threshold is assumed here.

