import heapq

def schedule_element_consumption(list_of_blocks, M):
    """
    编排元素进食顺序，以满足每个波次一个块最多贡献一个元素的约束。

    Args:
        list_of_blocks (list[list]): 包含N个块的列表，每个块是一个元素列表。
        M (int): 消费者数量。

    Returns:
        list[list]: 进食波次列表，每个波次是一个将被消耗的元素列表。
    """
    
    # 1. 初始化
    # next_index[i] 存储块 i 中下一个待消耗的元素的索引。
    N = len(list_of_blocks) # 块的数量
    next_index = [0] * N
    
    # 总的元素数量
    total_elements = sum(len(block) for block in list_of_blocks)
    
    # 结果列表：存储所有进食波次
    scheduled_waves = []
    
    # 2. 循环直到所有元素都被消耗
    elements_consumed = 0
    while elements_consumed < total_elements:
        
        # 当前波次将要消耗的元素列表
        current_wave = []
        
        # 3. 贪心选择：从 N 个块中选择元素
        # 约束：每个块最多贡献一个元素，总共不超过 M 个元素。
        
        # 遍历所有块
        for i in range(N):
            block = list_of_blocks[i]
            
            # 检查：
            # a) 该块是否还有未消耗的元素？ (next_index[i] < len(block))
            # b) 当前波次的元素数量是否已达到消费者上限 M？ (len(current_wave) < M)
            if next_index[i] < len(block) and len(current_wave) < M:
                
                # 贪心选择：选择该块的下一个元素
                element_to_consume = block[next_index[i]]
                current_wave.append(element_to_consume)
                
                # 更新块的进度
                next_index[i] += 1
        
        # 4. 记录波次结果
        if current_wave:
            scheduled_waves.append(current_wave)
            elements_consumed += len(current_wave)
        else:
            # 理论上不会发生，除非 total_elements 计算有误
            break 
            
    return scheduled_waves


def schedule_element_consumption_optimized(list_of_blocks, M):
    """
    使用最大堆优化：优先消费剩余元素最多的块，以最小化总波次。
    """
    # 1. 初始化堆
    # Python的heapq是最小堆，所以我们将长度取负数来实现最大堆
    # 存储结构: (-剩余长度, 原始索引, 当前元素指针, 块引用)
    # 加入原始索引是为了在长度相同时保持稳定性
    pq = []
    for i, block in enumerate(list_of_blocks):
        if block: # 只添加非空块
            heapq.heappush(pq, (-len(block), i, 0, block))
    
    scheduled_waves = []
    
    while pq:
        current_wave = []
        temp_storage = [] # 用于暂存本轮取出的块
        
        # 2. 贪心填充当前波次
        # 从堆中取出最多 M 个剩余最长的块
        while pq and len(current_wave) < M:
            neg_len, idx, ptr, block = heapq.heappop(pq)
            
            # 消费该块的下一个元素
            current_wave.append(block[ptr])
            
            # 更新该块的状态
            remaining = -neg_len - 1
            if remaining > 0:
                # 稍后放回堆中，指针+1
                temp_storage.append((-remaining, idx, ptr + 1, block))
        
        # 3. 将本轮消费过的块放回堆中
        for item in temp_storage:
            heapq.heappush(pq, item)
            
        scheduled_waves.append(current_wave)
            
    return scheduled_waves

from collections import deque

def schedule_element_consumption_fast_rr(list_of_blocks, M):
    """
    使用队列优化：跳过已空的块，提高运行速度，但保持轮询逻辑。
    """
    # 1. 初始化队列
    # 存储结构: [块内当前索引, 块的数据引用]
    # 使用 deque 实现高效的移除和添加
    queue = deque()
    for block in list_of_blocks:
        if block:
            queue.append([0, block]) # [index, block_data]
            
    scheduled_waves = []
    
    while queue:
        current_wave = []
        blocks_processed_this_wave = 0
        
        # 本轮循环的上限是当前队列长度（避免同一波次重复取同一个块）
        # 或者是 M（消费者上限）
        count_to_check = min(len(queue), M)
        
        for _ in range(count_to_check):
            # 取出队头
            curr_idx, block = queue.popleft()
            
            # 消费元素
            current_wave.append(block[curr_idx])
            
            # 检查是否还有剩余
            if curr_idx + 1 < len(block):
                # 如果还有，放回队尾（等待下一波）
                queue.append([curr_idx + 1, block])
            else:
                # 如果没有了，直接丢弃，不再放回队列
                pass
        
        # 注意：这里有一个逻辑变动。
        # 原始代码是每次波次都从头(index 0)开始扫描。
        # 使用队列会导致顺序变成 Round Robin (循环)。
        # 如果必须严格每一波都优先从第1个块尝试，队列法需要修改为每波重置迭代器，
        # 但那样不如直接用双指针法高效。
        
        scheduled_waves.append(current_wave)
        
    return scheduled_waves


# if __name__ == "__main__":
#     # --- 示例 ---
#     blocks = [
#         ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'],  
#         ['B1', 'B2'],              
#         ['C1', 'C2', 'C3'],        
#         ['D1', 'D2'],                     
#         ['E1', 'E2', 'E3', 'E4'],
#         ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
#         ["G1"],
#         ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8" ]
#     ]
#     M_consumers = 2 # 消费者数量

#     schedule = schedule_element_consumption(blocks, M_consumers)

#     ## 5. 结果输出
#     print(f"--- 原始块数量 N: {len(blocks)}, 消费者数量 M: {M_consumers} ---")
#     for i, wave in enumerate(schedule):
#         print(f"波次 {i+1} ({len(wave)} 元素): {wave}")

#     print("\n--- 调度分析 ---")
#     print(f"总波次数: {len(schedule)}\n")

#     # schedule2 = schedule_element_consumption_optimized(blocks, M_consumers)

#     # ## 5. 结果输出
#     # print(f"--- 原始块数量 N: {len(blocks)}, 消费者数量 M: {M_consumers} ---")
#     # for i, wave in enumerate(schedule2):
#     #     print(f"波次 {i+1} ({len(wave)} 元素): {wave}")

#     # print("\n--- 调度分析 ---")
#     # print(f"总波次数: {len(schedule2)}\n")


#     # schedule3 = schedule_element_consumption_fast_rr(blocks, M_consumers)

#     # ## 5. 结果输出
#     # print(f"--- 原始块数量 N: {len(blocks)}, 消费者数量 M: {M_consumers} ---")
#     # for i, wave in enumerate(schedule3):
#     #     print(f"波次 {i+1} ({len(wave)} 元素): {wave}")

#     # print("\n--- 调度分析 ---")
#     # print(f"总波次数: {len(schedule3)}\n")
