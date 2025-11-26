cd .

python -m start_training \
    --model_path_to_load Qwen/Qwen3-4B \
    --training_type 'cpt' \
    --checkpoint_epoch '0,1,2' \
    --pack_length 1024 \
    --use_NLIRG 'true' \
    --epoch 2 \
    --data_path data/cpt_example.json \
    --output_dir outputdir \
    --batch_size 2 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
