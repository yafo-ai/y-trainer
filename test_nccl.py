import os
import torch
import torch.distributed as dist

# 检查 CUDA 可用性
if not torch.cuda.is_available():
    print("Error: CUDA is not available. This test requires GPUs.")
    exit()

def test_nccl_communication(rank, world_size):
    """
    初始化进程组，并在所有 GPU 之间执行 NCCL All-Reduce 操作。
    """
    
    # 1. 初始化进程组
    # 'nccl' 后端用于 GPU 上的高效通信
    try:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize process group with NCCL. Error: {e}")
        return

    # 2. 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)
    
    print(f"Rank {rank}/{world_size}: Process group initialized on GPU {rank}.")

    # 3. 创建测试数据
    # 每个 GPU 创建一个张量，初始值为它的 rank + 1
    # 例如：rank 0 -> [1.0], rank 1 -> [2.0], rank 2 -> [3.0]
    initial_value = float(rank + 1)
    tensor = torch.tensor([initial_value], dtype=torch.float32).cuda(rank)

    # 预期总和 (Sum of ranks + 1): 1 + 2 + 3 + ... + world_size
    expected_sum = world_size * (world_size + 1) / 2.0
    
    print(f"Rank {rank}: Initial tensor value: {tensor.item():.1f}")
    
    # 4. 执行 NCCL All-Reduce
    # 使用 dist.all_reduce()，操作是 SUM (求和)
    # 这会高效地将所有 GPU 上的张量相加，并将结果同步回所有 GPU
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # # 5. 验证结果
    # if torch.isclose(tensor.item(), torch.tensor(expected_sum).item()):
    #     status = "SUCCESS"
    # else:
    #     status = "FAILURE"

    print(f"--- Rank {rank} Result ---")
    print(f"  Final value after All-Reduce: {tensor.item():.1f}")
    print(f"  Expected sum: {expected_sum:.1f}")
    # print(f"  NCCL Communication Test: {status}")
    print("----------------------------\n")

    # 6. 清理
    # dist.destroy_process_group()

if __name__ == '__main__':
    # -----------------------------------------------------------
    # !!! 关键步骤 !!!
    # Python 分布式应用需要一个专用的启动器来管理多进程和环境变量。
    # 您需要通过 `torchrun` 或 `mpirun` 来运行此脚本，而不是直接运行它。
    # -----------------------------------------------------------
    
    # 自动获取环境变量 (由 torchrun 设置)
    WORLD_SIZE = int(os.environ.get('WORLD_SIZE', -1))
    RANK = int(os.environ.get('RANK', -1))

    if WORLD_SIZE > 0 and RANK >= 0:
        # 如果通过 torchrun 启动
        test_nccl_communication(RANK, WORLD_SIZE)
    else:
        # 如果直接运行此脚本，则提供启动说明
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            print(f"Detected {num_gpus} GPU(s). This test requires at least 2 GPUs.")
            print("Please ensure your environment is set up correctly.")
        
        print("\n--- 启动说明 ---")
        print("请保存此代码（例如为 `nccl_test.py`），然后使用 `torchrun` 启动：")
        print(f"torchrun --nproc_per_node={min(2, num_gpus)} nccl_test.py")
        print(f"（这里假设您想测试 {min(2, num_gpus)} 张卡）")
        print("-------------------")