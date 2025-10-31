cd .

deepspeed --master_port 29501 --include localhost:0,1 --module training_code.start_training \
    --model_path_to_load model_or_path \
    --training_type 'sft' \
    --use_deepspeed \
    --use_NLIRG \
    --data_path example_dataset/sft_example.json \
    --output_dir outputdir \
    --batch_size 1 \
    --token_batch 10 \
    --use_lora \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
