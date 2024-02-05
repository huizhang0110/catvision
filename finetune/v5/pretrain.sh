#!/bin/bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=$(bash ib_stat.sh)
NODE_RANK=$(bash rank.sh)
echo $NCCL_IB_HCA
echo $NODE_RANK
DIR=`pwd`
GPUS_PER_NODE=8
NNODES=8
MASTER_ADDR=172.28.4.10
MASTER_PORT=6009
MODEL=/mnt/data/zhanghui/chat-mm-docs-pretrain-qwenvl/output-ckpts/pretrain/eva-glu-qwen14b/vitg_adapter_llm72b
DATA="/mnt/data/zhanghui/img2dataset_downloaded/sharegpt4v_pretrain.json"
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune_v2.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --train_llm False \
    --train_vllm False \
    --train_adapter True \
    --train_vit False \
    --output_dir /mnt/data/zhanghui/output-ckpts/lvlm/v5/pretrain \
    --max_steps 3000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --warmup_steps 200 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --deepspeed finetune/ds_config_zero3.json \
    --lazy_preprocess True \
    --resume_from_checkpoint /mnt/data/zhanghui/output-ckpts/lvlm/v5/pretrain/checkpoint-500