# cd .

python -m start_training \
    --model_path_to_load /root/autodl-fs/model/test_model_qwen1.5b \
    --training_type 'sft' \
    --epoch 3 \
    --checkpoint_epoch '0,1' \
    --data_path data/sft_example.json \
    --output_dir outputdir \
    --batch_size 1 \
    --token_batch 10 \
    --use_NLIRG 'true' \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
