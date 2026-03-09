import os
import re
from typing import Dict, List


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


class XinsongRAG:
    def __init__(self, kb_path: str, chunk_size: int = 600, chunk_overlap: int = 80):
        self.kb_path = kb_path
        self.chunk_size = max(200, int(chunk_size or 600))
        self.chunk_overlap = max(0, int(chunk_overlap or 80))
        self._chunks: List[Dict] = []
        self._loaded = False

    def _split_into_sections(self, text: str) -> List[Dict]:
        lines = text.splitlines()
        sections: List[Dict] = []
        current_title = "文档"
        buf: List[str] = []

        def flush_section() -> None:
            nonlocal buf
            content = "\n".join(buf).strip()
            if content:
                sections.append({"title": current_title, "content": content})
            buf = []

        for raw_line in lines:
            line = raw_line.rstrip()
            if line.startswith("#"):
                flush_section()
                current_title = line.lstrip("#").strip() or "文档"
            else:
                buf.append(line)

        flush_section()
        return sections

    def _chunk_section(self, title: str, content: str) -> List[Dict]:
        chunks: List[Dict] = []
        text = re.sub(r"\n{3,}", "\n\n", content).strip()
        if not text:
            return chunks

        start = 0
        total = len(text)
        chunk_id = 0

        while start < total:
            end = min(total, start + self.chunk_size)
            piece = text[start:end]
            if end < total:
                split_pos = max(piece.rfind("。"), piece.rfind("\n"), piece.rfind("；"), piece.rfind("!"), piece.rfind("?"))
                if split_pos > int(self.chunk_size * 0.5):
                    end = start + split_pos + 1
                    piece = text[start:end]

            piece = piece.strip()
            if piece:
                chunks.append(
                    {
                        "id": f"{title}-{chunk_id}",
                        "title": title,
                        "text": piece,
                        "norm": _normalize_text(piece),
                    }
                )
                chunk_id += 1

            if end >= total:
                break
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def load(self) -> None:
        if self._loaded:
            return
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"知识库文件不存在: {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            text = f.read()

        sections = self._split_into_sections(text)
        chunks: List[Dict] = []
        for sec in sections:
            chunks.extend(self._chunk_section(sec["title"], sec["content"]))

        self._chunks = chunks
        self._loaded = True

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        self.load()
        q = _normalize_text(query)
        if not q:
            return []

        terms = [t for t in re.split(r"[^\w\u4e00-\u9fff]+", q) if t]
        if not terms:
            terms = [q]

        scored = []
        for chunk in self._chunks:
            score = 0
            norm = chunk["norm"]
            for t in terms:
                if not t:
                    continue
                if t in norm:
                    score += 3
            if "新松" in norm:
                score += 1
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = max(1, int(top_k or 4))

        result = []
        for score, chunk in scored[:top_k]:
            result.append(
                {
                    "text": chunk["text"],
                    "source": chunk["title"],
                    "score": score,
                }
            )
        return result
