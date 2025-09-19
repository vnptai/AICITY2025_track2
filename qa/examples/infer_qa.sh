MODEL="OpenGVLab/InternVL3-78B" #CHECKPOINT_PATH

lmdeploy serve api_server $MODEL --chat-template internvl2_5 --server-port 23333 --tp 8

python run_vqa.py \
  --test_image_dir "test_processed_anno" \
  --test_file_dir "test" \
  --output_file "data/vqa_result.json" \
  --tmp_file "data/tmp/vqa_result" \
  --openai_api_key "" \ #LMDEPLOY_API_KEY
  --openai_api_base "" \ $#LMDEPLOY_API_URL
  --model "" \ #LMDEPLOY_MODEL
  --num_processes 32