import os
from rag_system import LegalRAG
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-1.5B-Instruct")
SFT_MODEL_PATH = os.path.join(BASE_DIR, "output", "sft", "legal_sft")
DB_PATH = os.path.join(BASE_DIR, "data", "knowledge", "law_db.txt")

# 1. 初始化 RAG 和模型
rag = LegalRAG(DB_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, SFT_MODEL_PATH).merge_and_unload()


def rag_chat(question):
    # 检索法条
    relevant_docs = rag.retrieve(question)
    context = "\n".join(relevant_docs)

    # 构造带知识的 Prompt
    prompt = f"<|im_start|>system\n你是一个法律专家。请根据以下参考法律条文回答问题：\n{context}\n<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.2)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# 测试一下
# 将 main_rag.py 最后改成这样
if __name__ == "__main__":
    print("\n--- 法律 AI 助手已启动（输入 'quit' 退出） ---")
    while True:
        user_input = input("【你】：")
        if user_input.lower() == 'quit': break
        answer = rag_chat(user_input)
        print(f"【助手】：{answer}")