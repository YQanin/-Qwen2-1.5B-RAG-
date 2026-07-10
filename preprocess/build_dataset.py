import os
import json
import pandas as pd

# ==========================
# 路径
# ==========================

CORPUS_DIR = "../data/corpus"
INSTRUCTION_DIR = "../data/instruction"

os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(INSTRUCTION_DIR, exist_ok=True)

pretrain_txt = os.path.join(
    CORPUS_DIR,
    "pretrain.txt"
)

pretrain_jsonl = os.path.join(
    CORPUS_DIR,
    "pretrain.jsonl"
)

# ==========================
# TXT -> JSONL
# ==========================

samples = []

with open(
        pretrain_txt,
        "r",
        encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if len(line) < 30:
            continue

        samples.append({
            "text": line
        })

with open(
        pretrain_jsonl,
        "w",
        encoding="utf-8"
) as f:

    for item in samples:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
        )

        f.write("\n")

print("JSONL生成完成！")

# ==========================
# 统计信息
# ==========================

statistics = []

lengths = []

for item in samples:

    length = len(item["text"])

    lengths.append(length)

statistics.append({

    "样本数量":

        len(samples),

    "平均长度":

        round(sum(lengths)/len(lengths),2),

    "最长样本":

        max(lengths),

    "最短样本":

        min(lengths)

})

df = pd.DataFrame(statistics)

df.to_csv(

    os.path.join(
        CORPUS_DIR,
        "corpus_statistics.csv"
    ),

    index=False,

    encoding="utf-8-sig"

)

print(df)

# ==========================
# Instruction模板
# ==========================

instruction_template = {

    "instruction":

        "请根据我国现行法律回答问题。",

    "input":

        "",

    "output":

        ""

}

with open(

    os.path.join(

        INSTRUCTION_DIR,

        "instruction_template.json"

    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        instruction_template,

        f,

        ensure_ascii=False,

        indent=4

    )

# ==========================
# 示例数据
# ==========================

example = [

    {

        "instruction":

            "什么是民事行为能力？",

        "input":

            "",

        "output":

            "民事行为能力是自然人能够通过自己的行为依法取得民事权利并承担民事义务的资格。"

    },

    {

        "instruction":

            "盗窃罪如何处罚？",

        "input":

            "",

        "output":

            "盗窃公私财物，数额较大的，依法追究刑事责任，并根据刑法规定量刑。"

    }

]

with open(

    os.path.join(

        INSTRUCTION_DIR,

        "instruction_example.json"

    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        example,

        f,

        ensure_ascii=False,

        indent=4

    )

print("Instruction模板生成完成！")