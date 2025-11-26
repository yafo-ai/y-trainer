import random
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import pandas as pd
from sklearn.decomposition import PCA

from src.ext.utils import get_vllm_embedding
from src.models.ext_response import ClustersResponse


def y_cluster(data):
    
    _array = np.array(data)
    
    # 检查是否有有效数据
    n_samples = len(_array)

    # 使用UMAP降维（HDBSCAN推荐预处理）或PCA作为替代
    n_components = min(30, n_samples)  # 确保不超过样本数

    pca = PCA(n_components=min(n_components, _array.shape[1]))
    core_data=pca.fit_transform(_array)

    # 使用HDBSCAN聚类 动态调整最小聚类大小(最小聚类，而不是聚类总数)
    min_cluster_size = max(2, min(5, n_samples//20))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,#一个簇至少要有多少个点
        min_samples=1,
        gen_min_span_tree=True,
        cluster_selection_method='eom'
    )
    clusters = clusterer.fit_predict(core_data)
    
    # 统计聚类信息
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    noise_points = np.sum(clusters == -1)
    print(f"识别到 {n_clusters} 个聚类，{noise_points} 个噪声点")

    return clusters


if __name__ == "__main__":
    
    data=[
    "用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：连续能打印多长时间",
    "用户正在浏览：爱普生L3256三合一彩色喷墨机（A4）[产品id:5A2655]用户：3253呢",
"用户正在浏览：爱普生L11058墨仓式彩色喷墨打印机（A3）[产品id:5A2679]用户：这个能自动打双面吗？",
"用户正在浏览：爱普生L11058墨仓式彩色喷墨打印机（A3）[产品id:5A2679]用户：自动双面吗？",
"用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：连续打印？",
"用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：用啥耗材",
"用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：使用什么耗材",
"用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：适用什么耗材？",
"用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：耗材型号"
  ]
    
    data_embedding = [get_vllm_embedding(text) for text in data]




    clusters = y_cluster(data_embedding)


    # 根据聚类信息将文本分组
    cluster_dict = {}
    for text, cluster in zip(data, clusters):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(text)

    # 从每个组中随机选择一个样本
    random_samples = [random.choice(cluster_dict[cluster]) for cluster in cluster_dict]
    
    response=ClustersResponse(
        clusters=clusters,
        data=data,
        random_samples=random_samples
    )

    print(response)

