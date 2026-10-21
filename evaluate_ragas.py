import os
from pathlib import Path
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import ChatOllama
from ragas.testset import TestsetGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "qwen2.5:7b")


def generate_evaluation_testset(pdf_path: str, testset_size: int = 5):
    """Generates synthetic RAG evaluation test set using Docling, Ollama, and RAGAS."""
    print(f"Parsing document: {pdf_path}...")
    converter = DocumentConverter()
    document = converter.convert(pdf_path)
    markdown_text = document.document.export_to_markdown()

    headers_to_split = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
    docs = markdown_splitter.split_text(markdown_text)
    print(f"Loaded {len(docs)} document chunks into memory.")

    print(f"Initializing Ollama ({OLLAMA_EVAL_MODEL}) and HuggingFace Embeddings...")
    ollama_model = ChatOllama(
        model=OLLAMA_EVAL_MODEL,
        temperature=0.0,
        format="json",
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Setting up RAGAS TestsetGenerator...")
    test_cases_generator = TestsetGenerator.from_langchain(
        llm=ollama_model,
        embedding_model=embeddings,
    )

    print(f"Generating test set of size {testset_size}...")
    test_set = test_cases_generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=testset_size,
    )

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ragas_testset.csv"

    try:
        df = test_set.to_pandas()
        df.to_csv(csv_path, index=False)
        print(f"Saved generated test set to: {csv_path}")
    except Exception:
        print("Test set generated:")
        print(test_set)

    return test_set


def main():
    default_doc = PROJECT_ROOT / "data" / "1706.03762v7.pdf"
    prompt_msg = f"ENTER YOUR FILE PATH (press Enter for default: {default_doc.name}): "
    user_input_path = input(prompt_msg).strip()

    pdf_file_path = user_input_path if user_input_path else str(default_doc)
    if not os.path.exists(pdf_file_path):
        print(f"Error: File '{pdf_file_path}' does not exist.")
        return

    generate_evaluation_testset(pdf_file_path, testset_size=5)


if __name__ == "__main__":
    main()
