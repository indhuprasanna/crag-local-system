from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi


# 1. Sample corpus containing both semantic ideas and specific technical IDs
raw_knowledge_base = [
    "Error code ERR-8092 occurs when the PostgreSQL connection pool exceeds maximum configured capacity.",
    "System authentication relies on RS256 signed JSON Web Tokens (JWT) with a 15-minute expiration window.",
    "Company holiday policy grants 20 days of paid annual leave plus federal public holidays.",
    "Data retention guidelines require audit trail logs to be persisted in cold S3 storage for a minimum of 7 years.",
    "Network timeout incidents during checkout should be reported under ticket tag INC-PAY-504."
]

# Convert strings into Document objects
initial_docs = [Document(page_content=text, metadata={"doc_id": i}) for i, text in enumerate(raw_knowledge_base)]

# 2. Chunking with overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
chunked_docs = text_splitter.split_documents(initial_docs)

print(f"Total chunks indexed: {len(chunked_docs)}")


# 3. Dense Vector Index (Chroma)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunked_docs, embeddings)


# 4. Sparse Keyword Index (BM25)
tokenized_corpus = [doc.page_content.lower().split() for doc in chunked_docs]
bm25 = BM25Okapi(tokenized_corpus)


# 5. Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(
    vector_results: List[Document], 
    bm25_results: List[Document], 
    k: int = 60, 
    top_n: int = 2
) -> List[Document]:
    """Combines rankings from two independent search retrieval systems."""
    rrf_score: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # Score vector search results
    for rank, doc in enumerate(vector_results):
        content = doc.page_content
        doc_map[content] = doc
        rrf_score[content] = rrf_score.get(content, 0.0) + (1.0 / (rank + 1 + k))

    # Score BM25 keyword results
    for rank, doc in enumerate(bm25_results):
        content = doc.page_content
        doc_map[content] = doc
        rrf_score[content] = rrf_score.get(content, 0.0) + (1.0 / (rank + 1 + k))

    # Sort descending by fused score
    sorted_docs = sorted(rrf_score.items(), key=lambda item: item[1], reverse=True)
    return [doc_map[content] for content, _ in sorted_docs[:top_n]]


def hybrid_search(query: str, top_n: int = 2) -> List[Document]:
    # 1. Fetch dense vector candidates
    dense_candidates = vectorstore.similarity_search(query, k=3)

    # 2. Fetch sparse BM25 candidates
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:3]
    sparse_candidates = [chunked_docs[i] for i in top_bm25_indices if bm25_scores[i] > 0]

    # 3. Fuse rankings
    fused_results = reciprocal_rank_fusion(dense_candidates, sparse_candidates, top_n=top_n)
    return fused_results


if __name__ == "__main__":
    print("\n--- Test 1: Exact Technical ID Search (BM25 excels here) ---")
    query_1 = "What causes ERR-8092?"
    results_1 = hybrid_search(query_1)
    for i, doc in enumerate(results_1, 1):
        print(f"Rank {i}: {doc.page_content}")

    print("\n--- Test 2: Semantic Conceptual Search (Vector excels here) ---")
    query_2 = "How long do we save audit trails?"
    results_2 = hybrid_search(query_2)
    for i, doc in enumerate(results_2, 1):
        print(f"Rank {i}: {doc.page_content}")