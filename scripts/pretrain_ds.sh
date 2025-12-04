cd .

deepspeed --master_port 29501 --include localhost:0,1 --module start_training \
    --model_path_to_load Qwen/Qwen3-4B \
    --training_type 'cpt' \
    --checkpoint_epoch '0,1,2' \
    --batch_size 2 \
    --use_deepspeed 'true'\
    --pack_length 1024 \
    --data_path data/cpt_example.json \
    --output_dir outputdir \
    --epoch 3 \
    --use_NLIRG 'true' \
    --enable_gradit_checkpoing 'true' \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
