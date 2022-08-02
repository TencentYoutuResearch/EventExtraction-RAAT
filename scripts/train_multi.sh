#! /bin/bash

NUM_GPUS=$1
PRETRAINED_MODEL_DIR=$2
shift

python3 -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} run_dee_task.py \
--exp_dir ./Exps \
--task_name RAAT \
--gradient_accumulation_steps 8 \
--train_batch_size 64 \
--use_bert True \
--bert_model ${PRETRAINED_MODEL_DIR} \
--re_label_map_path ./Data/label_map.json \
--logging_steps 100 \
--eval_batch_size 32 \
--learning_rate 5e-5 \
--data_dir ./Data \
--num_train_epochs 100 \
--save_cpt_flag True \
--raat True \
--raat_path_mem True \
--num_relation 18
