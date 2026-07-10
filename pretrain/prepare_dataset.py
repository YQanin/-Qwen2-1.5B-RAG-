import os
from datasets import load_dataset
from transformers import AutoTokenizer
from config import *

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
# Qwen2 没有官方 pad_token，使用 eos_token 替代
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    "json",
    data_files=DATA_PATH
)


def tokenize_function(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    # 将 input_ids 复制给 labels
    labels = result["input_ids"].copy()

    # 核心修复：将 padding 部分的 label 设为 -100，避免计算 Loss
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

    result["labels"] = labels
    return result


dataset = dataset.map(
    tokenize_function,
    batched=False,  # 关闭 batched 以便列表推导式正常工作
    remove_columns=["text"]
)

dataset.save_to_disk(DATASET_SAVE_PATH)
print(f"数据集已成功处理并保存至：{DATASET_SAVE_PATH}")
print(dataset)