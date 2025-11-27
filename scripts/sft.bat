@echo off
chcp 65001 >nul

echo Starting SFT Training...

python -m start_training ^
    --model_path_to_load "E:\GPT\LLM_models\Qwen2.5-0.5B-Instruct" ^
    --training_type "sft" ^
    --epoch 3 ^
    --use_NLIRG "true" ^
    --checkpoint_epoch "0,1" ^
    --data_path ".\data\sft_example.json" ^
    --output_dir ".\LLM_models\TEST_SFT" ^
    --batch_size 1 ^
    --token_batch 10 ^
    --enable_gradit_checkpoing 'true' \ 
    --use_lora 'true' \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

if %errorlevel% equ 0 (
    echo Training completed successfully!
) else (
    echo Training failed with error code %errorlevel%
    pause
)