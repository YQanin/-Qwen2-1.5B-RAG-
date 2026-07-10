# Legal-AI-Assistant: 基于 Qwen2.5 的法律垂直领域大模型 (全量代码库)

本项目是一个完整的法律大模型微调与 RAG 系统。本仓库包含代码、配置文件、实验数据集以及训练后的 LoRA 权重。

## 1. 核心目录说明
E:\PythonProject6\
├── .idea/            # IDE 工程配置文件 (PyCharm 相关)！未上传
├── data/             # 数据集存储目录 (包含 instruction/ 指令集与 knowledge/ 法律知识库)
├── figures/          # 训练过程可视化目录 (包含 loss 曲线图等实验结果)
├── finetune/         # 指令微调 (SFT) 核心代码实现
├── models/           # 基座大模型存储目录 (存放 Qwen2.5-1.5B-Instruct 等模型文件)！未上传
├── output/           # 训练产出目录 (存放预训练与 SFT 阶段的 LoRA 权重)
├── preprocess/       # 数据预处理脚本 (包含数据清洗、Tokenization 逻辑)
├── pretrain/         # 继续预训练 (CPT) 阶段代码实现
└── rag/              # 检索增强生成系统 (包含向量检索与 RAG 对话接口)

## 2. 环境部署
本项目依赖 `torch`, `transformers`, `peft`, `faiss-cpu`, `sentence-transformers` 等库。
```bash
pip install -r requirements.txt
