import os

# 动态获取项目根目录 E:\PythonProject6 (假设 config.py 在 pretrain 文件夹下)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-1.5B-Instruct")
DATA_PATH = os.path.join(BASE_DIR, "data", "corpus", "pretrain.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "pretrain")
DATASET_SAVE_PATH = os.path.join(OUTPUT_DIR, "tokenized_dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练超参数
MAX_LENGTH = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
EPOCHS = 1
LEARNING_RATE = 2e-5