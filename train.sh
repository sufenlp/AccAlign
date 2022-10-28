#!/bin/sh


WorkLOC=/home/acc_align #yours




TRAIN_FILE_SRC=$WorkLOC/xxx/train.src
TRAIN_FILE_TGT=$WorkLOC/xxx/train.tgt
TRAIN_FILE_ALIGN=$WorkLOC/xxx/train.talp

EVAL_FILE_SRC=$WorkLOC/xxx/dev.src
EVAL_FILE_TGT=$WorkLOC/xxx/dev.tgt
Eval_gold_file=$WorkLOC/xxx/dev.talp

OUTPUT_DIR_ADAPTER=$WorkLOC/adapter_output
Model=$WorkLOC/models/LaBSE

EVAL_RES=$WorkLOC/xxx/eval_result


CUDA_VISIBLE_DEVICES=0 python $WorkLOC/github_open/train_alignment_adapter.py \
    --output_dir_adapter $OUTPUT_DIR_ADAPTER \
    --eval_res_dir $EVAL_RES \
    --model_name_or_path $Model \
    --extraction 'softmax' \
    --train_so \
    --do_train \
    --do_eval \
    --train_data_file_src $TRAIN_FILE_SRC \
    --train_data_file_tgt $TRAIN_FILE_TGT \
    --eval_data_file_src $EVAL_FILE_SRC \
    --eval_data_file_tgt $EVAL_FILE_TGT \
    --per_gpu_train_batch_size 40 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 20 \
    --learning_rate 1e-4 \
    --save_steps 100  \
    --max_steps 1200 \
    --align_layer 6 \
    --logging_steps 50 \
    --eval_gold_file $Eval_gold_file \
    --gold_one_index \
    --softmax_threshold 0.1 \
    --train_gold_file $TRAIN_FILE_ALIGN \
    #--output_dir $OUTPUT_DIR \

exit

