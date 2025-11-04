cd /root/autodl-fs/auto-fs/code/testcode/y-trainer

python -m training_code.utils.schdule.sort \
    --data_path example_dataset/sft_example.json \
    --output_path example_dataset/sft_example_out.json \
    --model_path Qwen3/Qwen3-8B \
    --mode "similarity_rank"
