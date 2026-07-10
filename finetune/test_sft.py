import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 路径配置
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-1.5B-Instruct")
CPT_LORA_PATH = os.path.join(BASE_DIR, "output", "pretrain", "legal_cpt")
SFT_LORA_PATH = os.path.join(BASE_DIR, "output", "sft", "legal_sft")

print("1. 加载基座模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("2. 合并 CPT 阶段的法律知识...")
model = PeftModel.from_pretrained(base_model, CPT_LORA_PATH)
model = model.merge_and_unload()

print("3. 挂载 SFT 阶段的对话指令适配器...")
model = PeftModel.from_pretrained(model, SFT_LORA_PATH)
model.eval()

# 测试用例
test_questions = [
    "什么是民事行为能力？",
    "如果我在路上捡到别人丢失的手表，可以据为己有吗？"
]

print("\n================ 开始测试 ================\n")
for q in test_questions:
    # 严格按照 SFT 训练时的格式构建 Prompt
    prompt = f"<|im_start|>system\n请根据我国现行法律回答问题。<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,  # 法律问答需要严谨，温度调低
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    # 截取模型生成的新内容
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"【用户】：{q}")
    print(f"【模型】：{response}\n")
    print("-" * 50 + "\n")