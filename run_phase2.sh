#!/bin/bash
# Phase 2 Optimization - Quick Start Command
# Reduces training time from 790 hours to ~10-12 hours

python experiments/compare_quantum_vs_classical_dualdrug.py \
    --n_cases 200 \
    --sampling_interval 5 \
    --offline_epochs 50 \
    --batch_size 512 \
    --num_workers 8 \
    --cql_warmup_epochs 10 \
    --online_episodes 200 \
    --use_cql \
    --bc_weight 0.8 \
    --cql_alpha 1.0

# Expected results:
# - Data loading: ~72K transitions (80% reduction)
# - Training time: ~10-12 hours (98.5% reduction from 790h)
# - Patient diversity maintained with 200 cases
