import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="Document Viewer & RAG Interface",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document & Markdown Viewer")
st.caption("Upload and preview Markdown, Text, or converted document outputs.")

uploaded_file = st.file_uploader(
    "Choose a file to preview",
    type=["md", "txt"],
    help="Upload a Markdown (.md) or Text (.txt) file to view formatted content.",
)

if uploaded_file is not None:
    try:
        content = uploaded_file.read().decode("utf-8", errors="replace")
        st.success(f"Loaded '{uploaded_file.name}' successfully!")
        
        tab_preview, tab_raw = st.tabs(["Rendered Markdown", "Raw Content"])
        with tab_preview:
            st.markdown(content)
        with tab_raw:
            st.code(content, language="markdown")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    # Check if there is a parsed document in outputs
    default_markdown = PROJECT_ROOT / "outputs" / "parsed_document.md"
    if default_markdown.exists():
        st.info(f"Showing default output from `{default_markdown.name}`:")
        with open(default_markdown, "r", encoding="utf-8") as f:
            preview_content = f.read()
        st.markdown(preview_content)
