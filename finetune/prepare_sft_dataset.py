import json
from datasets import Dataset
from transformers import AutoTokenizer
from finetune.config_sft import *

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

raw_data = []
for path in DATA_PATHS:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 如果是列表（如 instruction_example.json）
            if isinstance(data, list):
                raw_data.extend(data)
            # 如果是单个字典（如 instruction_template.json）
            elif isinstance(data, dict):
                raw_data.append(data)
    else:
        print(f"警告: 找不到文件 {path}")

# 核心过滤：剔除 output 为空的数据（模板文件没有答案，不能参与 Loss 计算）
valid_data = [item for item in raw_data if item.get("output", "").strip() != ""]

# 将清洗后的数据转化为 Hugging Face Dataset 格式
dataset = Dataset.from_list(valid_data)
print(f"成功加载有效 SFT 数据条数: {len(dataset)}")

def process_func(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    # 组合用户查询
    query = instruction + ("\n" + input_text if input_text else "")

    # 按照 Qwen 的 ChatML 对话格式硬编码拼接
    prompt = f"<|im_start|>system\n请根据我国现行法律回答问题。<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    response = f"{output_text}<|im_end|>\n"

    # 分别编码
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    # 拼接完整的 input_ids
    input_ids = prompt_ids + response_ids

    # 核心：将 prompt 部分的标签设为 -100，只对 response 计算 Loss
    labels = [-100] * len(prompt_ids) + response_ids

    # 截断或填充到 MAX_LENGTH
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    else:
        pad_len = MAX_LENGTH - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

    return {"input_ids": input_ids, "labels": labels}


print("开始处理并进行 Label Masking...")
tokenized_dataset = dataset.map(
    process_func,
    batched=False,
    remove_columns=dataset.column_names
)

tokenized_dataset.save_to_disk(DATASET_SAVE_PATH)
print(f"SFT 数据集已成功处理并保存至：{DATASET_SAVE_PATH}")