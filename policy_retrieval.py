import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = Path("data/policies")
OUTPUT_DIR = Path("outputs")
CHUNKS_PATH = OUTPUT_DIR / "policy_chunks.json"

CHUNK_WORDS = 40
OVERLAP_WORDS = 10
STRIDE = CHUNK_WORDS - OVERLAP_WORDS


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []

    stride = chunk_words - overlap_words
    chunks = []

    for start in range(0, len(words), stride):
        chunk = " ".join(words[start:start + chunk_words])
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break

    return chunks


def load_policy_chunks():
    records = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_text(text), start=1):
            records.append({
                "source": path.name,
                "chunk_id": f"{path.stem}_{index}",
                "text": chunk,
            })

    return records


class PolicyRetriever:
    def __init__(self, records):
        if not records:
            raise ValueError(f"No policy chunks found in {POLICY_DIR.resolve()}")

        self.records = records
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(
            [record["text"] for record in records]
        )

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:min(k, len(self.records))]

        return [
            {
                "source": self.records[i]["source"],
                "score": round(float(scores[i]), 6),
                "text": self.records[i]["text"],
            }
            for i in ranked
        ]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_policy_chunks()

    CHUNKS_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    retriever = PolicyRetriever(records)

    print("Policy files:", len(list(POLICY_DIR.glob("*.txt"))))
    print("Total chunks:", len(records))
    print("Chunks by source:")

    for source in sorted({record["source"] for record in records}):
        count = sum(record["source"] == source for record in records)
        print(f"  {source}: {count}")

    print("Saved:", CHUNKS_PATH.resolve())

    query = "When can a customer receive a refund for a late delivery?"
    print(f"\nRetrieval test: {query}")

    for result in retriever.retrieve(query, k=3):
        print(
            f"\nSource: {result['source']} | "
            f"Score: {result['score']}"
        )
        print(result["text"])


if __name__ == "__main__":
    main()
