import os
from pathlib import Path
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Tracing and parallelism configuration
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ADVANCED_RAG")
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "Hybrid-RAG")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:latest")


def convert_pdf_to_markdown(pdf_path: str, output_path: str | None = None) -> str:
    """Converts a PDF document to markdown format using Docling."""
    converter = DocumentConverter()
    print(f"Converting document: {pdf_path}...")
    document = converter.convert(pdf_path)
    markdown_content = document.document.export_to_markdown()

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, mode="w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Markdown file saved to: {out_file}")

    return markdown_content


def build_vector_store(markdown_text: str):
    """Splits markdown into header-aware chunks and indexes into Qdrant."""
    headers_to_split = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
    docs = markdown_splitter.split_text(markdown_text)
    print(f"Created {len(docs)} document chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION,
        force_recreate=True,
    )
    print("Vector store indexing completed.")
    return vector_store


def run_rag_pipeline(pdf_path: str, user_query: str):
    """Executes the complete Advanced RAG workflow."""
    output_markdown_path = PROJECT_ROOT / "outputs" / "parsed_document.md"
    markdown_text = convert_pdf_to_markdown(pdf_path, str(output_markdown_path))

    # Build vector store
    vector_store = build_vector_store(markdown_text)
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    )

    # Initialize models
    ollama_model = ChatOllama(
        model=OLLAMA_MODEL_NAME,
        temperature=0.7,
        verbose=False,
    )

    google_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        max_output_tokens=1024,
        api_key=GEMINI_API_KEY,
    )

    # Pre-retrieval query rewriting agent
    print("Rewriting query for optimal retrieval...")
    output_parser = StrOutputParser()
    pre_retrieval_agent = create_agent(
        model=ollama_model,
        system_prompt=(
            "Re-write the user question converting it into a professional question "
            "which is used by an LLM in fetching the correct information. "
            "Just provide the rewritten query and no other text."
        ),
    )

    rewrite_response = pre_retrieval_agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]}
    )
    rewritten_query = output_parser.invoke(rewrite_response["messages"][-1].content).strip()
    print(f"Rewritten Query: {rewritten_query}\n")

    # Cross-encoder reranker
    print("Running Cross-Encoder Reranker...")
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=vector_retriever,
    )

    final_docs = compression_retriever.invoke(rewritten_query)
    print(f"Retrieved and reranked {len(final_docs)} chunks.")

    # Generate answer with Gemini
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a RAG system. Answer the user's question accurately using ONLY the provided context below.\n"
            "Structure your answer clearly and in detail. Do not hallucinate.\n"
            "If the context is insufficient or an error occurs, explain it in simple layman terms.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ])

    rag_chain = prompt | google_model | StrOutputParser()
    answer = rag_chain.invoke({
        "question": rewritten_query,
        "context": final_docs,
    })

    return answer


def main():
    default_doc = PROJECT_ROOT / "data" / "1706.03762v7.pdf"
    prompt_msg = f"ENTER YOUR FILE PATH (press Enter for default: {default_doc.name}): "
    user_input_path = input(prompt_msg).strip()

    pdf_file_path = user_input_path if user_input_path else str(default_doc)
    if not os.path.exists(pdf_file_path):
        print(f"Error: File '{pdf_file_path}' does not exist.")
        return

    user_query = input("\nENTER YOUR QUESTION / PROMPT:\n").strip()
    if not user_query:
        print("Error: Prompt cannot be empty.")
        return

    answer = run_rag_pipeline(pdf_file_path, user_query)
    print("\n" + "=" * 50)
    print("RAG ANSWER:")
    print("=" * 50)
    print(answer)


if __name__ == "__main__":
    main()
