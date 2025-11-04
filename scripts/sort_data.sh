cd /root/autodl-fs/auto-fs/code/testcode/y-trainer

python -m training_code.utils.schdule.sort \
    --data_path /root/autodl-fs/auto-fs/data/v73_sft_all_1012_4471.json \
    --output_path /root/autodl-fs/auto-fs/data/v73_sft_all_1012_4471_sorted.json \
    --model_path /root/autodl-fs/model/Qwen3-8B \
    --mode "filtered_rank"