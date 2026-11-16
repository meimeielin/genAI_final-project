"""
RAG 檢索模組（雛形版）
- 把 knowledge/ 底下的 Markdown 依「## 小節」切塊
- 用中文字元 bigram + TF-IDF + 餘弦相似度做檢索（不需額外套件、不需 API）
- 正式版會換成 Embedding 模型 + 向量資料庫（Chroma / pgvector），介面不變
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DISORDER_FILES = {
    "depression": "depression.md",
    "gad": "gad.md",
    "bipolar": "bipolar.md",
    "schizophrenia": "schizophrenia.md",
}


@dataclass
class Chunk:
    doc: str        # 檔名
    title: str      # 文件標題
    section: str    # 小節標題
    text: str       # 內容

    @property
    def cite(self) -> str:
        return f"{self.title} › {self.section}"


def _tokens(text: str) -> list[str]:
    text = re.sub(r"\s+", "", text.lower())
    words = re.findall(r"[a-z0-9]+", text)          # 英文單字
    han = re.sub(r"[^一-鿿]", "", text)      # 中文字
    bigrams = [han[i:i + 2] for i in range(len(han) - 1)]
    return words + bigrams + list(han)


class KnowledgeBase:
    def __init__(self, folder: str | Path = Path(__file__).parent / "knowledge"):
        self.chunks: list[Chunk] = []
        for path in sorted(Path(folder).glob("*.md")):
            self._load(path)
        self._index()

    def _load(self, path: Path):
        title, section, buf = path.stem, "概要", []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                if buf:
                    self.chunks.append(Chunk(path.name, title, section, "\n".join(buf).strip()))
                section, buf = line[3:].strip(), []
            elif line.strip():
                buf.append(line.strip())
        if buf:
            self.chunks.append(Chunk(path.name, title, section, "\n".join(buf).strip()))

    def _index(self):
        self.tf = [Counter(_tokens(c.title + c.section + c.text)) for c in self.chunks]
        df = Counter(t for tf in self.tf for t in tf)
        n = len(self.chunks)
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        self.vecs = [self._vec(tf) for tf in self.tf]

    def _vec(self, tf: Counter) -> dict:
        v = {t: c * self.idf.get(t, 0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {t: x / norm for t, x in v.items()}

    def search(self, query: str, k: int = 3, disorder: str | None = None) -> list[tuple[float, Chunk]]:
        q = self._vec(Counter(_tokens(query)))
        results = []
        for vec, chunk in zip(self.vecs, self.chunks):
            score = sum(w * vec.get(t, 0) for t, w in q.items())
            # 指定疾病時，加權該疾病文件與共通文件（認知扭曲、溝通原則）
            if disorder:
                if chunk.doc == DISORDER_FILES.get(disorder):
                    score += 0.15
                elif chunk.doc in DISORDER_FILES.values():
                    score -= 0.15
            results.append((score, chunk))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]


def format_context(hits: list[tuple[float, Chunk]]) -> str:
    return "\n\n".join(f"[{i + 1}] {c.cite}\n{c.text}" for i, (_, c) in enumerate(hits))
