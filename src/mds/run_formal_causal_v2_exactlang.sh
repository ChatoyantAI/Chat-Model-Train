#!/usr/bin/env bash
set -euo pipefail

root=/home/gyr/lusychat-training/mds-dialogue-rp-v1
cd "$root"

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

test -f 00_manifest/CAUSAL_FUSION_WEIGHT_DECISION.json
test ! -e 03_embeddings/fused_causal_v2_u745_r005_s25.f16.npy
test ! -e 00_manifest/CAUSAL_FUSED_U745_R005_S25_ASSEMBLY_REPORT.json
test ! -e 04_clusters/clusters_causal_v2_exactlang_k1000.parquet
test ! -e 04_clusters/clusters_causal_v2_exactlang_k2000.parquet
test ! -e 05_selection/selection_mmr_causal_v2_exactlang_k1000.parquet
test ! -e 05_selection/selection_mmr_causal_v2_exactlang_k2000.parquet

.venv/bin/python scripts/preflight_formal_causal_v2.py

.venv/bin/python scripts/assemble_fused_causal_v2.py \
  --root "$root" \
  --chunk-size 4096

.venv/bin/python scripts/cluster_rp_mds_causal_v2_exactlang.py \
  --root "$root" \
  --k-values 1000,2000 \
  --batch-size 4096 \
  --max-iter 100 \
  --n-init 3 \
  --max-no-improvement 20 \
  --seed 20260724 \
  --minimum-cluster-capacity 20 \
  --output-tag causal_v2_exactlang

.venv/bin/python scripts/select_rp_mds_mmr_causal_v2_exactlang.py \
  --root "$root" \
  --k 1000 \
  --lambda-value 0.5 \
  --seed 20260724 \
  --keep-all-small 5 \
  --output-tag causal_v2_exactlang

.venv/bin/python scripts/select_rp_mds_mmr_causal_v2_exactlang.py \
  --root "$root" \
  --k 2000 \
  --lambda-value 0.5 \
  --seed 20260724 \
  --keep-all-small 5 \
  --output-tag causal_v2_exactlang

echo "FORMAL_CAUSAL_V2_EXACTLANG_PIPELINE_COMPLETE"
