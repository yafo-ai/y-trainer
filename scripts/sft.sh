cd .

python -m training_code.start_training \
    --model_path_to_load model_or_path \
    --lora_path ./lora \
    --training_type 'sft' \
    --epoch 1 \
    --checkpoint_epoch '0,1,2' \
    --use_NLIRG \
    --data_path example_dataset/sft_example.json \
    --output_dir outputdir \
    --use_lora \
    --batch_size 1 \
    --token_batch 10 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
