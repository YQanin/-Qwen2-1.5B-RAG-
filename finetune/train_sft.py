import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from finetune.config_sft import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的计算设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("1. 正在加载基座模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("2. 正在合并 CPT 阶段的法律知识...")
# 加载第一阶段的 LoRA 并将其参数物理合并到主模型中
model = PeftModel.from_pretrained(base_model, CPT_LORA_PATH)
model = model.merge_and_unload()

# 核心修复：连通梯度计算图
model.enable_input_require_grads()

print("3. 正在初始化 SFT 阶段的新 LoRA 适配器...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16, # SFT 阶段稍微调大一点 rank，提升指令拟合能力
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

dataset = load_from_disk(DATASET_SAVE_PATH)
# 兼容处理：判断加载的是 Dataset 还是 DatasetDict
if hasattr(dataset, "column_names") and "input_ids" in dataset.column_names:
    train_dataset = dataset
else:
    train_dataset = dataset["train"]

# 使用 Seq2Seq 的 Collator 更适合 SFT
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    bf16=True,
    logging_steps=2, # 数据少的话，调小 logging_steps 方便看 Loss
    save_strategy="epoch",
    report_to="none",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    optim="adamw_torch",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

model.config.use_cache = False

print("开始 SFT 训练...")
trainer.train()

save_path = os.path.join(OUTPUT_DIR, "legal_sft")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"指令微调完成！最终的法律对话模型已保存至: {save_path}")