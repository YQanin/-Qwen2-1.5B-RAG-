import os
import re
from sentence_transformers import SentenceTransformer
import faiss

class LegalRAG:
    def __init__(self, db_path):
        # 使用多语言模型支持中文语义理解
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.db_path = db_path
        self.chunks = self._load_and_split()
        self.index, self.embeddings = self._build_index()

    def _load_and_split(self):
        # 使用正则表达式匹配【...】开头的内容，将整个法条提取为一个 chunk
        with open(self.db_path, "r", encoding="utf-8") as f:
            content = f.read()
            # 匹配模式：以【开头，直到下一个【或者结尾
            chunks = re.findall(r'【.*?】：.*?(?=\n【|$)', content, re.DOTALL)
            # 清理换行符，确保每个 chunk 是一条完整的法条
            return [c.replace('\n', '') for c in chunks if c.strip()]

    def _build_index(self):
        embeddings = self.model.encode(self.chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index, embeddings

    def retrieve(self, query, top_k=2):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        # 返回检索到的法条
        return [self.chunks[i] for i in indices[0]]