"""
embeddings.py
Handles chunking transcript text, generating vector embeddings,
storing them in ChromaDB, and retrieving relevant chunks for RAG.
"""

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"

RAG_PROMPT_TEMPLATE = """Answer the following question based only on the context below.

Context:
{context}

---

Question: {que}
"""


def generate_embeddings(text: str) -> Chroma:
    """
    Chunk the transcript, embed it with HuggingFace, and persist to ChromaDB.
    Returns the Chroma vector store instance.
    """
    doc = Document(page_content=text, metadata={"source": "youtube"})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents([doc])

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    return db


def closest(query: str, db: Chroma, k: int = 3):
    """Return the top-k most relevant chunks from the vector store."""
    results = db.similarity_search(query, k=k)
    if not results:
        print("No matching results found.")
        return []
    return results


def create_prompt(results: list, question: str) -> str:
    """Build a RAG prompt from retrieved chunks and the user's question."""
    if not results:
        return "Sorry, I couldn't find anything relevant."

    context_text = "\n\n---\n\n".join(
        doc.page_content if not isinstance(doc, tuple) else doc[0].page_content
        for doc in results
    )

    prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    return prompt_template.format(context=context_text, que=question)
