import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from src.ext.entropy import calculate_entropy
from src.ext.loss import calculate_loss
from src.ext.model_loader import ModelLoader
from src.train.data_loader import MultiFormatDataLoader

class AdvancedDivergenceAnalyzer:
    """
    Loss-Entropy 背离分析器
    特点：向量化计算、支持滑动窗口平滑、自动合并错误片段
    """
    
    def __init__(self, 
                 smooth_window: int = 3, 
                 z_score_threshold: float = 1.96): # 1.96 对应 95% 置信度
        """
        Args:
            smooth_window: 滑动窗口大小，用于平滑 Loss 和 Entropy 曲线，消除毛刺
            z_score_threshold: 异常判定的 Z-score 阈值
        """
        self.smooth_window = smooth_window
        self.z_threshold = z_score_threshold

    def analyze(self, 
                tokens: List[str], 
                losses: List[float], 
                entropies: List[float]) -> Dict[str, Any]:
        
        # 1. 构造 DataFrame 进行向量化处理
        df = pd.DataFrame({
            'token': tokens,
            'loss': losses,
            'entropy': entropies
        })

        # 2. 数据平滑 (关键改进：消除单点噪声)
        # 使用 rolling mean 平滑曲线，填充 NaN 为原始值
        df['loss_smooth'] = df['loss'].rolling(self.smooth_window, center=True, min_periods=1).mean()
        df['entropy_smooth'] = df['entropy'].rolling(self.smooth_window, center=True, min_periods=1).mean()

        # 3. 计算全局统计量 (基于平滑后的数据)
        loss_mean, loss_std = df['loss_smooth'].mean(), df['loss_smooth'].std()
        ent_mean, ent_std = df['entropy_smooth'].mean(), df['entropy_smooth'].std()
        
        # 防止除零
        loss_std = loss_std if loss_std > 1e-6 else 1.0
        ent_std = ent_std if ent_std > 1e-6 else 1.0

        # 4. 计算 Z-Score
        df['z_loss'] = (df['loss_smooth'] - loss_mean) / loss_std
        df['z_entropy'] = (df['entropy_smooth'] - ent_mean) / ent_std

        # 5. 计算核心指标：背离分数 (Divergence Score)
        # 逻辑：Loss 越高(正Z) 且 Entropy 越低(负Z)，分数越高
        # 公式：Z_loss - Z_entropy
        df['divergence_score'] = df['z_loss'] - df['z_entropy']

        # 6. 向量化分类 (核心逻辑优化)
        # 使用 np.select 替代循环，速度提升 100x
        conditions = [
            (df['z_loss'] > 0.6) & (df['z_entropy'] < -0.6), # 高Loss 低Entropy
            (df['z_loss'] > 0.6) & (df['z_entropy'] > 0.6),  # 高Loss 高Entropy
            (df['z_loss'] < -0.6) & (df['z_entropy'] > 0.6), # 低Loss 高Entropy
            (df['loss'] > loss_mean) & (df['z_entropy'] < -1.0) # 极度自信但Loss高于平均
        ]
        choices = [
            '幻觉/标注错误', # 盲目自信 (最危险：幻觉/标注错误)
            '困难样本',      # 困难样本 (不知道，且知道自己不知道)
            '运气好/平庸',         # 运气好/平庸 (可能是高频停用词)
            '可疑的确定性'         # 可疑的确定性
        ]
        
        df['pattern'] = np.select(conditions, choices, default='Normal')

        # 7. 提取高风险片段 (Span Extraction)
        # 仅仅找出 Token 不够，我们要把连在一起的 Token 合并成一个“错误片段”
        anomalies = self._extract_error_spans(df)
        
        # 8. 计算统计概览
        stats = df['pattern'].value_counts().to_dict()
        

        # 9.计算异常token的占比
        anomaly_score = df[df['pattern'] != 'Normal']['divergence_score'].abs().sum() / len(df)


        return {
            "anomaly_score":anomaly_score,
            "error_spans": anomalies,
            "statistics": stats,
            "metrics": {
                "avg_divergence": df['divergence_score'].mean(),
                "max_divergence": df['divergence_score'].max()
            },
            # 返回原始 df 的一部分用于 debug 或绘图
            "debug_data": df[['token', 'loss', 'entropy', 'divergence_score', 'pattern']].to_dict(orient='list')
        }

    def _extract_error_spans(self, df: pd.DataFrame) -> List[Dict]:
        """
        将连续的异常 Token 合并为片段，方便人类审核
        """
        # 过滤出非 Normal 的行
        mask = df['pattern'] != 'Normal'
        error_df = df[mask].copy()
        
        if error_df.empty:
            return []

        # 识别连续片段：如果 index 是连续的，归为一组
        # group_id 在索引不连续时会增加
        error_df['group_id'] = (error_df.index.to_series().diff() > 1).cumsum()
        
        spans = []
        for _, group in error_df.groupby('group_id'):
            # 只有当该组中最严重的背离分超过阈值时，才通过
            max_div = group['divergence_score'].max()
            if max_div < self.z_threshold:
                continue

            # 获取主要的问题类型 (众数)
            primary_pattern = group['pattern'].mode()[0]
            
            # 组合文本
            text_span = "".join(group['token'].tolist())
            
            spans.append({
                "text": text_span,
                "start_idx": int(group.index[0]),
                "end_idx": int(group.index[-1]),
                "avg_loss": float(group['loss'].mean()),
                "avg_entropy": float(group['entropy'].mean()),
                "divergence_score": float(max_div),
                "pattern": primary_pattern
            })
            
        # 按严重程度排序
        spans.sort(key=lambda x: x['divergence_score'], reverse=True)
        return spans


def loss_entropy_divergence_analyze(user_input: str, ai_output: str, model: Any, tokenizer: Any, smooth_window: int = 3, z_score_threshold: float = 1.96) -> Dict:
    """
    分析 Loss-Entropy 背离，返回异常片段和统计信息
    """
    analyzer = AdvancedDivergenceAnalyzer(smooth_window=smooth_window, z_score_threshold=z_score_threshold)
    output_tokens, losses= calculate_loss(user_input, ai_output, model, tokenizer)
    _,entropies=calculate_entropy(user_input, ai_output,model,tokenizer)
    result= analyzer.analyze(output_tokens, losses, entropies)

    return result


def loss_entropy_divergence_analyze_batch(data_path:str,output_path:str, model: Any, tokenizer: Any, smooth_window: int = 3, z_score_threshold: float = 1.96):
    """
    批量分析 Loss-Entropy 背离，返回异常片段和统计信息
    """

    loader = MultiFormatDataLoader(file_paths=data_path,skip_empty=True)
    datas = loader.load_data()

    try:
        for d in tqdm(datas):
            user_input = d.get('instruction','')+d.get('input','')
            ai_output = d.get('output','')
            result=loss_entropy_divergence_analyze(user_input, ai_output, model, tokenizer,smooth_window,z_score_threshold)
            d['loss_entropy_analyze_result']=result
    
        datas = sorted(datas, key=lambda x: x['loss_entropy_analyze_result']["anomaly_score"], reverse=True)

    except Exception as e:
        print(f"分析过程中出错: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in datas:
            json_str = json.dumps(entry, ensure_ascii=False)
            f.write(json_str + '\n')
    
    print("分析结束")



# if __name__ == "__main__":
#     print("开始语料质量分析...")
    
#     model_loader = ModelLoader("")
#     # model_loader.switch_model("/raid1/big-models/qwen2.5-7b-instruct/","")
#     model_loader.switch_model("E:\llm_model\Qwen2.5-0.5B-Instruct","")
#     tokenizer=model_loader.load_tokenizer()
#     model = model_loader.load_model()
    

#     user_input="\n你是一个专业的文档总结，请参考“知识库查询结果”中有用的信息，提取并总结和用户问题有关的知识\n\n当前时间：2024-12-11 16:55:27\n\n用户的聊天记录：\n\n用户: 正在浏览： 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】[产品id:2096137]\n用户: 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】[产品id:2096137]\n用户: 你好，重量多少，厚度多少\n\n知识库查询结果：\n*********\n\n# 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】\nNcCode 2096137\n存货编码： 2096137\n存货名称： 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】\n基础配置： 集显 Intel i7 Win11 （8GB+8GB+1TB） -1340P\n对内型号： G540 Gen2-078\n对外型号： 华为擎云 G540 Gen2\n上市时间： 2024年4月\n## 机身（尺寸、重量、材质、开合反转角度）、风扇\n机身重量： 约1.45kg\n机身材质： 外壳正面铝合金，屏幕边框/键盘触控板面/外壳底面塑胶\n翻转角度： 180度\n单手开合： 不支持\n风扇： 风扇×1\n机身尺寸（长x宽x厚）： 325mm x 218mm x 17.2mm\n\n\n# 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】\nNcCode 2096137\n存货编码： 2096137\n存货名称： 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】\n基础配置： 集显 Intel i7 Win11 （8GB+8GB+1TB） -1340P\n对内型号： G540 Gen2-078\n对外型号： 华为擎云 G540 Gen2\n上市时间： 2024年4月\n## CPU处理器、显卡、内存、硬盘、扩容性、SSD\n显卡类型： 集成显卡\nCPU类型： 第13代智能英特尔® 酷睿™ i7-1360P 处理器\nCPU核数： 12核：4性能核+8能效核\nCPU频率： 性能核心：基频2.2GHz，最高频率5.0GHz\n效率内核：基频1.6GHz，最高频率3.7GHz\nCPU线程数： 16线程\n运行内存容量/频率： 16GB/3200MHz\n运行内存类型： DDR4\n运行内存形态（RAM）： 板载内存+SO-DIMM插卡内存\n运行内存通道（RAM）： 双通道\n硬盘类型/容量： SSD固态/1TB\nSSD接口理论传输速率： PCIE3.0×4：32Gbps\nSSD协议/形态： NVMe/M.2 2280、NVMe/M.2 2242\n扩容： 内存和硬盘是支持扩容，显卡不支持\n硬盘位： 2个\n显卡： 集成显卡：英特尔® 锐炬® Xe显卡\n\n\n# 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】\nNcCode 2096137\n存货编码： 2096137\n存货名称： 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】\n基础配置： 集显 Intel i7 Win11 （8GB+8GB+1TB） -1340P\n对内型号： G540 Gen2-078\n对外型号： 华为擎云 G540 Gen2\n上市时间： 2024年4月\n## 节能环保认证（能效等级、Evo）\nEvo认证： 不支持\n中国能效等级认证： 1级能效\n\n\n# 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】\nNcCode 2096137\n存货编码： 2096137\n存货名称： 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】\n基础配置： 集显 Intel i7 Win11 （8GB+8GB+1TB） -1340P\n对内型号： G540 Gen2-078\n对外型号： 华为擎云 G540 Gen2\n上市时间： 2024年4月\n## 售后保修期限、售后服务\n服务： 全国联保：1500+华为授权服务中心，遍布全国300+城市（不含港澳台）\n便捷寄修：提供双向免费寄修服务，足不出户完成维修\n保修期限： 三年整机质保、三年上门\n*********\n\n\n\n用户的聊天记录：\n\n用户: 正在浏览： 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】[产品id:2096137]\n用户: 【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】[产品id:2096137]\n用户: 你好，重量多少，厚度多少\n\n总结文档中与用户的聊天记录有关的内容，注意要友好简洁，通俗易懂。总结回答时，浓缩用户的商品名，可以使用借代。注意：切记你是一个语气温和，善于站在用户角度理解问题的在线客服。\n\n\n\n\n\n\n\n\n\n\n\n\nterminate 工具介绍：用于将结果/回答发送给用户\nterminate 工具指令：command=|<|terminate(message=\"总结文档中与用户的聊天记录有关的内容，注意要友好简洁，通俗易懂\")|>|\n\n输出工具指令command，注意工具指令不可篡改，不可遗漏，必须以command=|<|开始，以|>|结束。"
#     ai_output="command=|<|terminate(message=\"您好，【定制惠普】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】的重量约为1.45kg，厚度为17.2mm。希望这些信息对您有帮助！\")|>|"
#     # ai_output="command=|<|terminate(message=\"华为擎云 G540 Gen2笔记本的重量是约1.45kg，厚度为17.2mm。\")|>|"
#     # ai_output="command=|<|terminate(message=\"您好，【定制】华为擎云 G540 Gen2笔记本(i7-1360P+16G+1TB)-Win11-【深空灰】的重量约为1.45kg，厚度为17.2mm。希望这些信息对您有帮助！\")|>|"
#     # ai_output="command=|<|terminate(message=\"您好，您浏览的华为擎云 G540 Gen2笔记本（i7-1360P+16G+1TB）重量约为1.45kg，厚度为17.2mm。如有其他问题，欢迎随时咨询！\")|>|"
#     # ai_output="command=|<|terminate(message=\"您好！您询问的这款定制笔记本重量约为1.45kg，厚度为17.2mm，属于轻薄商务本的范畴。铝合金外壳材质也兼顾了耐用性和便携性，适合日常通勤携带。\")|>|\n"

#     # user_input="帮我介绍下北京"
#     # ai_output="北京是中国最大的省会。"

#     # 计算注意力分数
#     output_tokens, losses= calculate_loss(user_input, ai_output, model, tokenizer)

#     output_tokens,entropies=calculate_entropy(user_input, ai_output,model,tokenizer)

#     analyzer = AdvancedDivergenceAnalyzer(smooth_window=1, z_score_threshold=1.5)
#     result = analyzer.analyze(output_tokens, losses, entropies)



#     # fig1 = plot_curves_with_divergences(result["debug_data"]["loss"], result["debug_data"]["entropy"], None, "Original Curves")
#     # fig1.savefig("original_curves.png", dpi=300, bbox_inches='tight')


#     # fig4 = plot_algorithm_comparison2(result["debug_data"]["loss"], result["debug_data"]["entropy"],result["debug_data"]["pattern"],result["debug_data"]["token"])
#     # fig4.savefig("dtw_algorithm_view.png", dpi=300, bbox_inches='tight')

#     # print("=== 统计概览 ===")
#     # print(result['statistics'])
#     # print("\n=== 发现的高危片段 ===")
#     # for span in result['error_spans']:
#     #     print(f"片段: '{span['text']}' | 类型: {span['pattern']} | 严重度: {span['divergence_score']:.2f}")

#     print("\n分析完成！")