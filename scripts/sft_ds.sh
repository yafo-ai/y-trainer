cd .

deepspeed --master_port 29501 --include localhost:0,1 --module start_training \
    --model_path_to_load Qwen/Qwen3-4B \
    --training_type 'sft' \
    --use_NLIRG 'true' \
    --use_deepspeed 'true'\
    --data_path data/sft_example.json \
    --output_dir outputdir \
    --batch_size 1 \
    --token_batch 10 \
    --use_lora 'true' \
    --epoch: 3 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
