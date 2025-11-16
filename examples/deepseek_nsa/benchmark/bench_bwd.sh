#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
seqlens=($((32*1024)) $((64*1024)) $((96*1024)) $((128*1024)))
# seqlens=($((32*1024)))

for seqlen in "${seqlens[@]}"; do
    echo "Running backward benchmark with sequence length: $seqlen"
    PYTHONPATH=./ python benchmark/benchmark_nsa_bwd.py --seq_len $seqlen --batch 1 --heads 4 --head_query 64 --selected_blocks 32 --block_size 64
done
