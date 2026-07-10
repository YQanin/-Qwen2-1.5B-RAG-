import os

# 获取项目根目录 (E:\PythonProject6)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 路径配置
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-1.5B-Instruct")
CPT_LORA_PATH = os.path.join(BASE_DIR, "output", "pretrain", "legal_cpt")

# 【修改点】：将数据路径改为列表，包含你的两个文件
DATA_PATHS = [
    os.path.join(BASE_DIR, "data", "instruction", "instruction_example.json"),
    os.path.join(BASE_DIR, "data", "instruction", "instruction_template.json")
]

OUTPUT_DIR = os.path.join(BASE_DIR, "output", "sft")
DATASET_SAVE_PATH = os.path.join(OUTPUT_DIR, "tokenized_sft_dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# SFT 训练超参数
MAX_LENGTH = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
EPOCHS = 3
LEARNING_RATE = 5e-5