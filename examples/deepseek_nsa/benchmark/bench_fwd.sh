#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
seqlens=($((32*1024)) $((64*1024)) $((96*1024)) $((128*1024)))
# seqlens=($((128*1024)))

for seqlen in "${seqlens[@]}"; do
    echo "Running benchmark with sequence length: $seqlen"
    PYTHONPATH=./ python benchmark/benchmark_nsa_fwd.py --seq_len $seqlen --batch 1 --heads 4 --head_query 64 --selected_blocks 32 --block_size 64
done