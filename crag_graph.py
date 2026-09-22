from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

from hybrid_retriever import hybrid_search


# --- 1. State Definition ---
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    web_fallback: bool
    generation: str
    retry_count: int


# --- 2. Initialize Models & Tools ---
llm = ChatOllama(model="llama3.1:8b", temperature=0)
web_search_tool = DuckDuckGoSearchRun()


# --- 3. Structured Output Evaluators ---

# Grader 1: Document Relevance
class GradeDocument(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question: 'yes' or 'no'"
    )

evaluator_llm = llm.with_structured_output(GradeDocument)

grader_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an evaluator assessing relevance of a retrieved document to a user question.\n"
        "If the document contains keywords, semantic meaning, or answers related to the question, grade it as 'yes'.\n"
        "If it is completely unrelated, grade it as 'no'."
    ),
    ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}")
])

doc_grader = grader_prompt | evaluator_llm


# Grader 2: Hallucination Grader
class GradeHallucinations(BaseModel):
    """Binary score for hallucination check in generation."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts: 'yes' or 'no'"
    )

hallucination_grader = grader_prompt | llm.with_structured_output(GradeHallucinations)

hallucination_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an evaluator assessing whether an LLM generation is grounded in / supported by a set of facts.\n"
        "Grade 'yes' if the answer is completely supported by the context. Grade 'no' if the answer introduces unsupported claims."
    ),
    ("human", "Set of facts:\n\n{documents}\n\nLLM Generation: {generation}")
])

hallucination_chain = hallucination_prompt | llm.with_structured_output(GradeHallucinations)


# --- 4. Node Definitions ---

def retrieve_node(state: GraphState) -> GraphState:
    print("\n--- [NODE: RETRIEVE] Running hybrid retrieval ---")
    question = state["question"]
    docs = hybrid_search(question, top_n=2)
    return {
        "documents": docs,
        "question": question,
        "web_fallback": False,
        "generation": "",
        "retry_count": state.get("retry_count", 0)
    }

def grade_documents_node(state: GraphState) -> GraphState:
    print("--- [NODE: GRADE DOCS] Assessing document relevance ---")
    question = state["question"]
    docs = state["documents"]

    filtered_docs = []
    web_fallback = False

    for doc in docs:
        grade = doc_grader.invoke({"question": question, "document": doc.page_content})
        if grade.binary_score.lower() == "yes":
            print("  -> Doc Status: RELEVANT")
            filtered_docs.append(doc)
        else:
            print("  -> Doc Status: NOT RELEVANT")

    if len(filtered_docs) == 0:
        print("  -> No relevant internal docs found. Switching to web fallback.")
        web_fallback = True

    return {
        "documents": filtered_docs,
        "question": question,
        "web_fallback": web_fallback,
        "generation": "",
        "retry_count": state.get("retry_count", 0)
    }

def web_search_node(state: GraphState) -> GraphState:
    print("--- [NODE: WEB SEARCH] Querying DuckDuckGo ---")
    question = state["question"]
    search_results = web_search_tool.invoke(question)
    
    web_doc = Document(
        page_content=str(search_results),
        metadata={"source": "duckduckgo"}
    )
    return {
        "documents": [web_doc],
        "question": question,
        "web_fallback": True,
        "generation": "",
        "retry_count": state.get("retry_count", 0)
    }

def generate_node(state: GraphState) -> GraphState:
    print("--- [NODE: GENERATE] Drafting grounded response ---")
    question = state["question"]
    docs = state["documents"]
    retries = state.get("retry_count", 0)

    context_str = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
    
    gen_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a technical assistant. Answer the user prompt strictly and solely using the provided context.\n"
            "If the answer cannot be found in the facts, state: 'Information not available in sources.'\n\n"
            "Context:\n{context}"
        ),
        ("human", "{question}")
    ])
    
    rag_chain = gen_prompt | llm
    response = rag_chain.invoke({"context": context_str, "question": question})

    return {
        "documents": docs,
        "question": question,
        "web_fallback": state["web_fallback"],
        "generation": response.content,
        "retry_count": retries + 1
    }


# --- 5. Conditional Routing ---

def decide_to_generate(state: GraphState) -> Literal["web_search", "generate"]:
    if state["web_fallback"]:
        return "web_search"
    return "generate"

def check_hallucination(state: GraphState) -> Literal["grounded", "not grounded", "max_retries"]:
    print("--- [ROUTER: CHECK HALLUCINATION] Verifying answer against source facts ---")
    docs = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)

    if retry_count > 2:
        print("  -> Max retries reached. Exiting to avoid infinite loop.")
        return "max_retries"

    context_str = "\n\n".join([d.page_content for d in docs])
    grade = hallucination_chain.invoke({"documents": context_str, "generation": generation})
    
    if grade.binary_score.lower() == "yes":
        print("  -> Decision: Answer is fully grounded in facts.")
        return "grounded"
    else:
        print("  -> Decision: Answer hallucinated or unsupported! Rerouting to regenerate.")
        return "not grounded"


# --- 6. Build Graph ---

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)
workflow.add_edge("web_search", "generate")

workflow.add_conditional_edges(
    "generate",
    check_hallucination,
    {
        "grounded": END,
        "not grounded": "generate",
        "max_retries": END
    }
)

crag_app = workflow.compile()


if __name__ == "__main__":
    query = "What is the token expiration window for system authentication?"
    print(f"\n==========================================")
    print(f"RUNNING CRAG PIPELINE FOR: '{query}'")
    print(f"==========================================")
    result = crag_app.invoke({"question": query, "retry_count": 0})
    print(f"\nFINAL VERIFIED ANSWER:\n{result['generation']}")