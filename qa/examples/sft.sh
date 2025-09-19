TRAIN_DATA="" #PATH_TO_TRAIN_SET
VAL_DATA="" #PATH_TO_VAL_SET
MODEL_NAME="OpenGVLab/InternVL3-78B-Instruct"


swift sft \
  --model $MODEL_NAME \
  --dataset $TRAIN_DATA \
  --num_train_epochs "3" \
  --per_device_train_batch_size "8" \
  --learning_rate "1e-4" \
  --gradient_accumulation_steps "16" \
  --save_steps "100" \
  --save_total_limit "5" \
  --logging_steps "1" \
  --max_length "4096" \
  --eval_strategy "epoch" \
  --per_device_eval_batch_size "4" \
  --eval_datasets $VAL_DATA \
  --eval_limit "10" \
  --do_train \
  --do_eval \
  --use_hf true \
  --train_type "qlora" \
  --lora_rank "8" \
  --lora_alpha "32" \
  --target_modules "all-linear" 