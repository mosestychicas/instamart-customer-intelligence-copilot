import json
import re
from pathlib import Path

import pandas as pd

from mock_llm import chat, reset
from policy_retrieval import PolicyRetriever, load_policy_chunks

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

QUESTIONS = [
    ("P1", "Can a cancelled order receive a refund?",
     ["cancelled", "no refund"]),
    ("P2", "What happens when a completed order arrives after 45 minutes?",
     ["more than 45 minutes", "full refund"]),
    ("P3", "Who approves refunds above INR 1000?",
     ["human support lead"]),
    ("P4", "How long does a customer have to request a refund?",
     ["7 days"]),
    ("P5", "How quickly must an unauthorized transaction ticket reach security?",
     ["security team", "2 hours"]),
]


def sentence_rank(question, chunks):
    """Rank policy sentences directly using TF-IDF cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = []
    for chunk in chunks:
        # Split each chunk into sentences and retain its source.
        for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"]):
            sentence = sentence.strip()
            if sentence:
                sentences.append({
                    "source": chunk["source"],
                    "text": sentence,
                })

    if not sentences:
        return []

    documents = [question] + [s["text"] for s in sentences]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()

    ranked = sorted(
        zip(sentences, scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "source": item["source"],
            "text": item["text"],
            "score": round(float(score), 6),
        }
        for item, score in ranked[:3]
    ]


def ask_model(question, context):
    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT: Answer only from the supplied policy context. "
                "If the answer is absent, state that it cannot be found."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}",
        },
    ]
    response = chat(messages)
    return (
        response.choices[0].message.content or "",
        response.usage.total_tokens,
    )


def main():
    retriever = PolicyRetriever(load_policy_chunks())
    results = []

    for qid, question, expected in QUESTIONS:
        # Keep the required top-3 chunk retrieval and use sentence ranking
        # to reduce irrelevant sentences in the final answer context.
        chunks = retriever.retrieve(question, k=3)
        ranked_sentences = sentence_rank(
            question,
            [
                {"source": chunk["source"], "text": chunk["text"]}
                for chunk in chunks
            ],
        )

        context = "\n".join(
            f"[{item['source']}] {item['text']}"
            for item in ranked_sentences
        )

        reset()
        answer, tokens = ask_model(question, context)

        # Zero-context control: verify that the model abstains without evidence.
        reset()
        control_answer, control_tokens = ask_model(
            question, "No relevant policy text was retrieved."
        )

        answer_lower = answer.lower()
        passed = all(term.lower() in answer_lower for term in expected)

        results.append({
            "question_id": qid,
            "question": question,
            "answer": answer,
            "sources": ";".join(
                dict.fromkeys(item["source"] for item in ranked_sentences)
            ),
            "retrieval_scores": json.dumps(
                [item["score"] for item in ranked_sentences]
            ),
            "tokens_used": tokens,
            "expected_terms": ";".join(expected),
            "expected_terms_found": passed,
            "zero_overlap_answer": control_answer,
            "zero_overlap_tokens": control_tokens,
        })

    df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "policy_assistant_eval.csv"
    df.to_csv(output_path, index=False)

    print("Evaluation questions:", len(df))
    print("Expected-term checks passed:", int(df["expected_terms_found"].sum()))
    print("Expected-term checks failed:", int((~df["expected_terms_found"]).sum()))

    for _, row in df.iterrows():
        print(f"\n{row['question_id']}: {row['question']}")
        print("Answer:", row["answer"])
        print("Sources:", row["sources"])
        print("Expected terms found:", row["expected_terms_found"])
        print("Zero-context answer:", row["zero_overlap_answer"])

    print("\nSaved:", output_path.resolve())


if __name__ == "__main__":
    main()
