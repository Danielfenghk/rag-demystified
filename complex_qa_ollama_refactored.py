# ================================
# 100% Ollama-compatible refactor
# ================================
# Key changes vs original:
# 1. NO OpenAI function-calling
# 2. NO Instructor / Pydantic schemas
# 3. Subquestions generated via plain-text JSON prompt
# 4. Robust fallback if model returns non-JSON
# 5. Single vector_retrieval implementation

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
import warnings
from pathlib import Path

from openai_utils import llm_call

warnings.filterwarnings("ignore")

# ---------------- Embedding + FAISS ----------------
embedder = SentenceTransformer("all-mpnet-base-v2")
dimension = 768
indexes = {}  # {doc_name: (faiss_index, sentences)}


def build_vector_stores(wiki_docs):
    for doc_name, text in wiki_docs.items():
        print(f"Creating vector store for {doc_name}...")
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        if not sentences:
            sentences = [text[i:i+500] for i in range(0, len(text), 500)]

        embeddings = embedder.encode(sentences, batch_size=32)
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype("float32"))
        indexes[doc_name] = (index, sentences)

    print("All vector stores ready.\n")


# ---------------- Retrieval ----------------

def vector_retrieval(llm_model, question, doc_name, top_k=3):
    if doc_name not in indexes:
        return f"No information for {doc_name}.", 0

    index, sentences = indexes[doc_name]
    q_emb = embedder.encode([question]).astype("float32")
    _, I = index.search(q_emb, top_k)

    context = "\n".join([sentences[i] for i in I[0] if i >= 0])

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present, say you don't know.

Question: {question}
Context:
{context}
Answer:
"""

    response, cost = llm_call(model=llm_model, user_prompt=prompt)
    return response.choices[0].message.content.strip(), cost


# ---------------- Subquestion Generation ----------------

def generate_subquestions_ollama(question, doc_names, llm_model):
    """
    Ollama-compatible subquestion generator.
    Returns list of {question, file_name, function}
    """

    prompt = f"""
You are an assistant that decomposes a question into subquestions.

Available documents:
{', '.join(doc_names)}

Rules:
- Use function "vector_retrieval" for fact questions
- Each subquestion must reference ONE document
- Output MUST be valid JSON only

JSON format:
{{
  "subquestions": [
    {{"question": "...", "file": "Toronto", "function": "vector_retrieval"}}
  ]
}}

User question: {question}
"""
    print("\n prompt in subquestions is {}", prompt)
    response, _ = llm_call(model=llm_model, user_prompt=prompt)
    raw = response.choices[0].message.content.strip()
    print(response)
    try:
        data = json.loads(raw)
        return data.get("subquestions", [])
    except Exception:
        # Fallback: treat whole question as one subquestion
        print("⚠️ Subquestion JSON parse failed. Using fallback.")
        return [
            {
                "question": question,
                "file": doc_names[0],
                "function": "vector_retrieval",
            }
        ]


# ---------------- Aggregation ----------------

def aggregate_answers(llm_model, question, answers):
    context = "\n".join(answers)

    prompt = f"""
Use the following information to answer the question.
If unsure, say you don't know.

Question: {question}
Information:
{context}
Answer:
"""
    print("\n prompt in aggregate_answers is {}", prompt)
    response, cost = llm_call(model=llm_model, user_prompt=prompt)
    print(response)
    return response.choices[0].message.content.strip(), cost


# ---------------- Wikipedia Loader ----------------

def load_wiki_pages(page_titles):
    headers = {"User-Agent": "Mozilla/5.0"}
    docs = {}

    Path("data").mkdir(exist_ok=True)

    for title in page_titles:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            },
            headers=headers,
            timeout=30,
        ).json()

        page = next(iter(r["query"]["pages"].values()))
        text = page.get("extract", "")[:10000]
        docs[title] = text

        with open(f"data/{title}.txt", "w", encoding="utf-8") as f:
            f.write(text)

    return docs


# ---------------- Main ----------------
if __name__ == "__main__":
    llm_model = "deepseek-r1:8b"  # Ollama
    doc_names = ["Toronto", "Chicago", "Houston", "Boston", "Atlanta"]

    wiki_docs = load_wiki_pages(doc_names)
    build_vector_stores(wiki_docs)

    while True:
        question = input("\nQuestion (type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        subqs = generate_subquestions_ollama(question, doc_names, llm_model)

        answers = []
        total_cost = 0

        for i, sq in enumerate(subqs, 1):
            print(f"\n🤔 Subquestion {i}: {sq['question']} [{sq['file']}]")
            ans, cost = vector_retrieval(llm_model, sq["question"], sq["file"])
            print(f"✅ {ans}")
            answers.append(ans)
            total_cost += cost

        final, cost = aggregate_answers(llm_model, question, answers)
        total_cost += cost

        print("\n🎯 Final Answer:")
        print(final)
        print(f"🤑 Total cost: ${total_cost:.4f}")
