@echo off
chcp 65001 >nul

echo Starting SFT Training...

python -m training_code.start_training ^
    --model_path_to_load "Qwen/Qwen3-4B" ^
    --training_type "sft" ^
    --epoch 3 ^
    --checkpoint_epoch "0,1,2" ^
    --data_path ".\example_dataset\sft_example.json" ^
    --output_dir ".\LLM_models\TEST_SFT" ^
    --use_lora ^
    --batch_size 1 ^
    --token_batch 10 ^
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

if %errorlevel% equ 0 (
    echo Training completed successfully!
) else (
    echo Training failed with error code %errorlevel%
    pause
)