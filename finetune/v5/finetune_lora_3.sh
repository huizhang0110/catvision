#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=$(bash ib_stat.sh)
NODE_RANK=$(bash rank.sh)
echo $NCCL_IB_HCA
echo $NODE_RANK
DIR=`pwd`

GPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=172.28.4.10
MASTER_PORT=5003
MODEL=/mnt/data/zhanghui/output-ckpts/lvlm/v5/finetune_lora_4/chat_2000
DATA="/mnt/data/zhanghui/img2dataset_downloaded/sharegpt4v_finetune.json,/mnt/data/zhanghui/img2dataset_downloaded/chartqa_trainval.json,/mnt/data/zhanghui/img2dataset_downloaded/synthdog-zh.json,/mnt/data/zhanghui/img2dataset_downloaded/synthdog-en.json,/mnt/data/zhanghui/img2dataset_downloaded/infographics_trainval.json,/mnt/data/zhanghui/img2dataset_downloaded/sp_docqa_trainval.json,/mnt/data/zhanghui/img2dataset_downloaded/socr.json,/mnt/data/zhanghui/img2dataset_downloaded/cc12m_gpt4_caption.json,/mnt/data/zhanghui/img2dataset_downloaded/region_visual_genome.json,/mnt/data/zhanghui/img2dataset_downloaded/visual_genome.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune_lora.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --fix_vit True \
    --output_dir /mnt/data/zhanghui/output-ckpts/lvlm/v5/finetune_lora_5 \
    --max_steps 3000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --adam_beta2 0.96 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --use_lora \
    --gradient_checkpointing \
    --deepspeed finetune/ds_config_zero3_optimizer.json
