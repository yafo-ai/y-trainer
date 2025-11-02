- [English](../README.md)
- [中文](README_zh.md)

---

# Y-Trainer

Y-Trainer 是一个LLM模型微调训练框架。

# 📊 核心优势：

- 📉 **精准对抗过拟合**： 专门优化，有效解决SFT中的过拟合难题。

- 🧩 **突破遗忘瓶颈**： 无需依赖通用语料，即可卓越地保留模型的泛化能力，守住核心竞争力的同时实现专项提升！🏆

  

该框架包含以下三个核心组件：

继续预训练（Continued Pre-training，简称 CPT）

指令微调（Supervised Fine-tuning，简称 SFT）

强化学习（Reinforcement Learning，简称 RL）[将**很快发布**需要配合Y-agent使用]。

## 持续预训练（Continued Pre-training）

支持多种模型预训练方法，能够高效利用训练数据，提升模型在特定领域的能力。

## 监督微调（Supervised Fine-tuning）

与传统 SFT 不同，我们采用了一套自研的训练方法，具备如下优势：

- 在尽可能保留基座模型能力的前提下，限制语料中错误知识的影响；
- 无需进行数据集平衡操作，实现快速收敛，同时几乎不会损害模型的原始能力。

## 强化学习（Reinforcement Learning）

基于 SFT 构建的全新强化学习框架，具有以下显著优势：

- **低资源需求：** 无需依赖参考模型、奖励模型、价值网络等组件，仅需合理设计奖励函数即可完成训练；
- **训练稳定：** 以高熵 token 作为分支节点，自动生成语料树，并通过内置聚类算法进行剪枝，确保充分的探索空间。结合自适应梯度计算策略，使得整个训练过程更加稳定可靠。

# Introduction

您既可以训练完整模型，也可以仅训练 LoRA 适配器。

同时支持在**单 GPU** 或**多 GPU** 环境下进行训练。

# Installation

```bash
cd Y-TRAINER

pip install -r requirements.txt
```

# Quick Start

您可以通过以下示例脚本轻松训练自己的模型。

## 单 GPU
```bash
# Continue pretraining
bash scripts/pretrain_ds.sh

# sft training
bash y-trainer/scripts/sft.sh
```
## 多GPU
```bash
# Continue pretraining
bash scripts/pretrain_ds.sh

# sft training
bash y-trainer/scripts/sft_ds.sh
```

## 训练数据格式简介

进行继续预训练时, 请参考 [此文件](../example_dataset/cpt_example.json)

```json
[
  {
    "ID": 0,
    "output": "your content 1"
  },
  {
    "ID": 1,
    "output": "your content 2"
  }
]
```

进行指令微调时, 请参考 [此文件](../example_dataset/sft_example.json)
```json
[
  {
    "id": 0,
    "instruction": "instruction 0",
    "output": "output 0",
    "input": ""
  },
  {
    "id": 1,
    "instruction": "instruction 1",
    "output": "output 1",
    "input": ""
  }
]
```

只有*output*字段的token会被训练。

更多教程请参见 [Y-Studio 官方文档](https://www.y-agent.cn/docs)。

# Y-Agent Studio 框架简介

**Y-Agent Studio** 框架**完全开源**、**可商用**。下载后即可**无限制**使用全部功能。

该框架融合了**代码的灵活性**与**可视化界面的便捷性**，支持以下能力：

- **流程编排与迭代**
- **自动化测试**
- **语料标注与生产管理**

## ✅ 主要特性

- 高度可定制的工作流，支持嵌套结构与有环的循环连接
- 完善的日志系统，提供可视化展示与自动化分析
- 开放的系统集成能力，可与您现有的 IT 系统无缝对接
- 支持自动化测试、语料标注、语料生产与管理
- 解决了垂直领域训练导致基座模型能力下降的问题

## 架构图（Architecture Diagram）

![架构图](system_architecture.webp)
