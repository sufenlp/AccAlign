#!/bin/sh


WorkLOC=/home/acc_align #yours

SRC=$WorkLOC/xxx/roen/roen.src
TGT=$WorkLOC/xxx/roen/roen.tgt

OUTPUT_DIR=$WorkLOC/xxx/infer_output
ADAPTER=$WorkLOC/xxx/adapter
Model=$WorkLOC/xxx/LaBSE



python $WorkLOC/github_open/aligner/train_alignment_adapter.py \
    --infer_path $OUTPUT_DIR \
    --adapter_path $ADAPTER \
    --model_name_or_path $Model \
    --extraction 'softmax' \
    --infer_data_file_src $SRC \
    --infer_data_file_tgt $TGT \
    --per_gpu_train_batch_size 40 \
    --gradient_accumulation_steps 1 \
    --align_layer 6 \
    --softmax_threshold 0.1 \
    --do_test \


exit


