#!/bin/bash

# Default parameters
GPU_LIST=${1:-0,1}            # Default GPU list
MASTER_ADDR=${2:-"localhost"} # Default localhost
MASTER_PORT=${3:-29650}       # Default port

echo "Starting DeepSpeed training..."
echo "Using GPUs: $GPU_LIST"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

# Set environment variables
export OMP_NUM_THREADS=1

deepspeed --include=localhost:$GPU_LIST \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          train_normal.py

echo "Training completed!"
