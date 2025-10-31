cd .

deepspeed --master_port 29501 --include localhost:0,1 --module code.start_training \
    --model_path_to_load model_or_path \
    --training_type 'cpt' \
    --checkpoint_epoch '0,1,2' \
    --batch_size 2 \
    --use_deepspeed \
    --pack_length 1024 \
    --use_NLIRG \
    --data_path example_dataset/cpt_example.json \
    --output_dir outputdir \
    --use_lora \
    --epoch 3 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
