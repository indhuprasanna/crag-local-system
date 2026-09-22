import sys
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

print("\n--- Step 1: Testing Local LLM (Llama 3.1) ---")
try:
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    response = llm.invoke("Reply in one short sentence: What is the main purpose of Corrective RAG?")
    print("LLM Response:\n", response.content.strip())
    print("\n[SUCCESS] Local LLM is operational.\n")
except Exception as e:
    print(f"\n[FAILED] LLM error: {e}")
    sys.exit(1)

print("--- Step 2: Testing Local Embeddings (Nomic Embed) ---")
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    sample_vector = embeddings.embed_query("Testing vector generation.")
    print(f"Generated embedding dimensions: {len(sample_vector)}")
    print("[SUCCESS] Local embeddings are operational.\n")
except Exception as e:
    print(f"\n[FAILED] Embeddings error: {e}")
    sys.exit(1)

print("--- Step 3: Testing ChromaDB Semantic Retrieval ---")
try:
    docs = [
        Document(page_content="CRAG uses a document grader to evaluate context relevance before generation."),
        Document(page_content="Photosynthesis is the process by which green plants make food from sunlight."),
        Document(page_content="Python is an interpreted, high-level, general-purpose programming language.")
    ]

    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    query = "How does Corrective RAG verify retrieved text?"
    retrieved_docs = retriever.invoke(query)

    print(f"Query: '{query}'")
    print(f"Retrieved Document:\n'{retrieved_docs[0].page_content}'")
    print("\n[SUCCESS] Vector store indexing and retrieval are operational.\n")
except Exception as e:
    print(f"\n[FAILED] Vector store error: {e}")
    sys.exit(1)

print("ALL SYSTEMS READY: Milestone 1 is verified!")