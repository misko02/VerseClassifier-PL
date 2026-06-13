from __future__ import annotations


def split_into_four_line_chunks(text: str) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    chunks: list[str] = []

    for start in range(0, len(lines), 4):
        chunk = lines[start : start + 4]
        if len(chunk) == 4:
            chunks.append("\n".join(chunk))

    return chunks
