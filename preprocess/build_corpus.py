import os
import re
import jieba
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 路径
CLEAN_DIR = "../data/clean"
CORPUS_DIR = "../data/corpus"
STAT_DIR = "../data/statistics"

os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(STAT_DIR, exist_ok=True)

corpus = []

sentence_length = []

word_counter = Counter()

# 遍历文件
for file in os.listdir(CLEAN_DIR):

    if not file.endswith(".txt"):
        continue

    path = os.path.join(CLEAN_DIR, file)

    with open(path, "r", encoding="utf-8") as f:

        text = f.read()

    # 按法律条文切分
    articles = re.split(r"(?=第.{1,5}条)", text)

    for article in articles:

        article = article.strip()

        if len(article) < 30:
            continue

        corpus.append(article)

        sentence_length.append(len(article))

        words = jieba.lcut(article)

        word_counter.update(words)

# 保存Corpus
corpus_path = os.path.join(
    CORPUS_DIR,
    "pretrain.txt"
)

with open(
        corpus_path,
        "w",
        encoding="utf-8"
) as f:
    for item in corpus:
        f.write(item)
        f.write("\n")
print("Corpus构建完成")

# 长度统计
df = pd.DataFrame({
    "length": sentence_length
})

df.to_csv(
    os.path.join(STAT_DIR,
                 "sentence_length.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 长度分布
plt.figure(figsize=(8,5))

plt.hist(
    sentence_length,
    bins=30
)

plt.xlabel("条文长度")

plt.ylabel("数量")

plt.title("法律条文长度分布")

plt.tight_layout()

plt.savefig(
    os.path.join(STAT_DIR,
                 "length_distribution.png"),
    dpi=300
)

plt.close()

# Top20高频词
stop_words = {
    "的","和","是","应当","不得","可以","以及",
    "以上","以下","本法","国家","规定"
}

top_words = [
    (k,v)
    for k,v in word_counter.items()
    if k not in stop_words
    and len(k)>1
]

top_words = sorted(
    top_words,
    key=lambda x:x[1],
    reverse=True
)[:20]

df2 = pd.DataFrame(
    top_words,
    columns=["词语","频率"]
)

df2.to_csv(
    os.path.join(STAT_DIR,
                 "top20_words.csv"),
    index=False,
    encoding="utf-8-sig"
)


# 高频词柱状图
plt.figure(figsize=(10,6))

plt.bar(

    df2["词语"],

    df2["频率"]

)

plt.xticks(rotation=45)

plt.title("Top20高频词")

plt.tight_layout()

plt.savefig(
    os.path.join(STAT_DIR,
                 "top20_words.png"),
    dpi=300

)

plt.close()

# 词云
wc = WordCloud(
    font_path="C:/Windows/Fonts/simhei.ttf",
    width=1000,
    height=700,
    background_color="white"
)

wc.generate_from_frequencies(
    dict(top_words)
)

wc.to_file(
    os.path.join(STAT_DIR,
                 "wordcloud.png")
)
print("统计完成")