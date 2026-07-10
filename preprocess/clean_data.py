import os
import re
import chardet
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 路径
RAW_DIR = "../data/raw/law"
CLEAN_DIR = "../data/clean"
STAT_DIR = "../data/statistics"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(STAT_DIR, exist_ok=True)

# 自动识别编码
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]

# 清洗函数

def clean_text(text):
    # 去除多余空格
    text = re.sub(r"[ \t]+", " ", text)

    # 去掉空行
    text = re.sub(r"\n+", "\n", text)

    # 去掉全角空格
    text = text.replace("\u3000", "")

    # 去掉页码
    text = re.sub(r"第\s*\d+\s*页", "", text)

    # 去掉多个连续空格
    text = re.sub(r" +", " ", text)

    return text.strip()

# 主程序
statistics = []

for file in os.listdir(RAW_DIR):

    if not file.endswith(".txt"):
        continue

    path = os.path.join(RAW_DIR, file)

    encoding = detect_encoding(path)

    print(f"{file} 编码：{encoding}")

    with open(path, "r", encoding=encoding, errors="ignore") as f:
        text = f.read()

    raw_chars = len(text)

    cleaned = clean_text(text)

    clean_chars = len(cleaned)

    lines = cleaned.split("\n")

    # 去重复行
    unique_lines = list(dict.fromkeys(lines))

    cleaned = "\n".join(unique_lines)

    with open(
        os.path.join(CLEAN_DIR, file),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(cleaned)

    statistics.append({

        "文件":
            file,

        "原始字符数":
            raw_chars,

        "清洗后字符数":
            clean_chars,

        "最终行数":
            len(unique_lines)

    })

print("全部清洗完成！")

# 保存统计
df = pd.DataFrame(statistics)

df.to_csv(
    os.path.join(STAT_DIR, "statistics.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(df)

# 字符数量统计图
plt.figure(figsize=(10,6))

plt.bar(
    df["文件"],
    df["清洗后字符数"]
)

plt.xticks(rotation=30)

plt.ylabel("字符数")

plt.title("法律文本字符数量统计")

plt.tight_layout()

plt.savefig(
    os.path.join(STAT_DIR,"char_count.png"),
    dpi=300
)

plt.close()

# 行数统计图
plt.figure(figsize=(10,6))

plt.bar(
    df["文件"],
    df["最终行数"]
)

plt.xticks(rotation=30)

plt.ylabel("行数")

plt.title("法律文本行数统计")

plt.tight_layout()

plt.savefig(
    os.path.join(STAT_DIR,"line_count.png"),
    dpi=300
)

plt.close()
print("统计图生成完成！")