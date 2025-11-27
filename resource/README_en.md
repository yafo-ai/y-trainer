

- [中文](../README.md)
- [English](README_en.md)

---

# Y-Trainer
Welcome to Y-Trainer—an advanced training framework designed for LLM fine-tuning.

# 🎯 Core Advantages:

- 📉 **Precision Targeting of Overfitting**: Expertly optimized to effectively resolve overfitting challenges in SFT.

- 🧩 **Breakthrough the Forgetting Bottleneck**: Excels at preserving the model's generalization capabilities without relying on massive general corpora. It maintains core competencies while enabling specialized enhancement! 🏆

**Schedule:** Reinforcement Learning (RL) will be released **soon**.

## Continued Pre-training
Supports model pre-training methods, enabling efficient utilization of training data to improve the model's capabilities in specified domains.

## Supervised Fine-tuning
Unlike traditional SFT, we employ a proprietary training method that achieves the following effects:

- Limits the influence of incorrect knowledge in the corpus while preserving the base model's capabilities as much as possible.
- Eliminates the need for dataset balancing, enabling fast convergence while almost never compromising the model's original capabilities.

## Reinforcement Learning
A brand-new reinforcement learning framework based on SFT, with the following advantages:

- **Low resource requirements:** No need for reference models, reward models, value networks, etc. Training can be completed by properly designing a reward function.
- **Stable training:** Uses high-entropy tokens as branch nodes to automatically generate a corpus tree, then employs built-in clustering algorithms for pruning to ensure sufficient exploration. Combined with adaptive gradient calculation, the training process is stable and reliable.


# Introduction

You can train a full model or just a LoRA adapter

You can also train models in **single gpu** or **multi - GPUs**

# Installation
```bash
cd Y-TRAINER
pip install -r requirements.txt
```

# Quick Start

Start the Training Service
You can start the training service by running the following command. (Before starting, you can modify the port and specify the large model folder in configs/server_config.py).

```bash
python main.py
```
Then, open your browser and navigate to:
http://localhost:8010

You can now use this professional AI text processing tool, which provides convenient features for training, integrated attention analysis, content compression, sample generation, and clustering-based filtering.


You can easily use these scripts to train your own model. 

Train the model in single GPU or multi - GPUs by following example scripts.

## Single GPU
```bash
# Continue pretraining
bash y-trainer/scripts/pretrain_ds.sh

# sft training
bash y-trainer/scripts/sft.sh
```
## Multi - GPUs
```bash
# Continue pretraining
bash y-trainer/scripts/pretrain_ds.sh

# sft training
bash y-trainer/scripts/sft_ds.sh
```

## Training data description

For cpt, see the json file in [cpt dataset example path](example_dataset/cpt_example.json)

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

For SFT, see the json file in [cpt dataset example path](example_dataset/sft_example.json)
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

*output* token will be trained only.

For more tutorials see Y-Studio Document [Y-Studio Document url](https://www.y-agent.cn/docs)


# Y-Agent Studio Framework

The **Y-Agent Studio** framework is **fully open-source**, **commercially usable**, and **does not differentiate between community and commercial editions**. Once downloaded, you gain access to **all features without restriction**.

It combines the **flexibility of coding** with the **convenience of a visual interface**, enabling:

- **Process orchestration and iteration**
- **Automated testing**
- **Corpus annotation and production/management**


## ✅ Features

- Highly customizable workflow, supporting nesting and cyclic (looped) connections
- Comprehensive logging system, with visual representation and automated analysis
- Open system integration capabilities, allowing seamless integration with your existing IT infrastructure
- Automated testing, corpus annotation, and corpus production/management
- The issue where vertical-domain training degrades base model capabilities

## Architecture Diagram
![Architecture Diagram](resource/system_architecture.webp)

