"""cs_agent.rag — 知识库检索（v0.5 内存版）

v0.5 定位：先把"检索 → 拼上下文 → 模型基于资料作答"这条链路跑通，
用轻量中文分词 + 关键词重叠打分。v1 换成 pgvector + embedding，本模块
对外接口不变（retrieve(query, top_k) -> list[Chunk]），替换点单一。

为什么先做内存版？验收标准是"能引用知识库回答"，裁判是链路通不通，
不是向量库强不强。工程上讲：先跑通再优化，避免过早引入重型依赖。

公开接口：
    KnowledgeBase.load(dir)      —— 加载目录下所有 .md，按标题切块
    kb.retrieve(query, top_k)    —— 返回最相关的若干知识块
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    source: str      # 来源文件
    title: str       # 所属小节标题
    text: str        # 正文
    score: float = 0.0

    def cite(self) -> str:
        return f"[{self.source} · {self.title}]"


def _tokenize(text: str) -> set[str]:
    """极简中文分词：按非中文/非字母数字切分 + 中文双字滑窗。

    够用即可：v0.5 只求链路正确，不求检索质量。v1 换 embedding 后
    这层会被彻底替换，所以不值得在这里投入复杂度。
    """
    tokens: set[str] = set()
    # 英文/数字词
    for w in re.findall(r"[A-Za-z0-9\-]+", text.lower()):
        tokens.add(w)
    # 中文：连续中文串切双字滑窗（"退款政策" -> 退款/款政/政策）
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seg) == 1:
            tokens.add(seg)
        for i in range(len(seg) - 1):
            tokens.add(seg[i:i + 2])
    return tokens


class KnowledgeBase:
    """从本地 markdown 加载知识，按二级标题（##）切块。"""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    @classmethod
    def load(cls, directory: str | Path) -> "KnowledgeBase":
        directory = Path(directory)
        chunks: list[Chunk] = []
        if not directory.exists():
            return cls([])
        for md in sorted(directory.glob("*.md")):
            chunks.extend(cls._split_markdown(md))
        return cls(chunks)

    @staticmethod
    def _split_markdown(path: Path) -> list[Chunk]:
        text = path.read_text(encoding="utf-8")
        chunks: list[Chunk] = []
        current_title = path.stem
        buffer: list[str] = []

        def flush():
            body = "\n".join(buffer).strip()
            if body:
                chunks.append(Chunk(source=path.name, title=current_title, text=body))

        for line in text.splitlines():
            if line.startswith("## "):
                flush()
                buffer = []
                current_title = line[3:].strip()
            elif line.startswith("# "):
                continue  # 一级标题作为文件主题，跳过
            else:
                buffer.append(line)
        flush()
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        """关键词重叠打分（Jaccard 变体），返回 TopK。"""
        if not self.chunks:
            return []
        q = _tokenize(query)
        if not q:
            return self.chunks[:top_k]
        scored: list[Chunk] = []
        for c in self.chunks:
            ct = _tokenize(c.title + " " + c.text)
            if not ct:
                continue
            overlap = len(q & ct)
            # 标题命中加权：标题是主题浓缩，命中更重要
            title_bonus = 2 * len(q & _tokenize(c.title))
            score = (overlap + title_bonus) / (len(q) ** 0.5)
            if score > 0:
                scored.append(Chunk(c.source, c.title, c.text, score))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def as_context(self, query: str, top_k: int = 3) -> str:
        """把 TopK 拼成给模型的上下文文本（带引用标注）。"""
        hits = self.retrieve(query, top_k)
        if not hits:
            return ""
        return "\n\n".join(f"{h.cite()}\n{h.text}" for h in hits)
