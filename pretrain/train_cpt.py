import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from config import *
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的计算设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 模型加载：使用 bfloat16 防溢出，并为 LoRA 准备
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16, # RTX 4060 支持 bf16
    device_map="auto",
    trust_remote_code=True
)

model.enable_input_require_grads()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 直接使用绝对路径加载
dataset = load_from_disk(DATASET_SAVE_PATH)
train_dataset = dataset["train"]

# mlm=False 表示这是自回归生成任务
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    bf16=True, # 使用 BF16 替代 FP16
    logging_steps=5, # 加快日志输出节奏，方便观察 Loss
    save_strategy="epoch",
    report_to="none", # 避免找不到 tensorboard 报错
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="adamw_torch",
    dataloader_num_workers=0, # Windows 下设为 0 避免多进程报错
    gradient_checkpointing=True, # 核心操作：显存节省神器！用计算换显存
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# 解决 gradient checkpointing 和 LoRA 一起使用的警告
model.config.use_cache = False

print("开始训练...")
trainer.train()


def save_loss_plot(trainer, output_dir):
    # 提取训练日志
    log_history = trainer.state.log_history
    steps = []
    losses = []

    for log in log_history:
        if "loss" in log:
            steps.append(log["step"])
            losses.append(log["loss"])

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Training Loss", color='blue', alpha=0.7)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 保存路径
    plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(plot_path)
    print(f"Loss 曲线图已保存至: {plot_path}")


# 调用保存函数
save_loss_plot(trainer, OUTPUT_DIR)
# ==========================================

save_path = os.path.join(OUTPUT_DIR, "legal_cpt")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"继续预训练完成！模型已保存至: {save_path}")