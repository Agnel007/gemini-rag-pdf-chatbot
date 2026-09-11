import os
import hashlib
import html

import streamlit as st
import fitz

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

        .main {
            padding-top: 1.5rem;
        }

        .title {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #777;
            margin-bottom: 30px;
        }

        .source-card {
            padding: 14px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
            background: #fafafa;
        }

        .source-title {
            font-weight: 600;
            margin-bottom: 8px;
        }

        .source-text {
            color: #666;
            font-size: 14px;
            line-height: 1.6;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document_key" not in st.session_state:
    st.session_state.active_document_key = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">📄 PDF AI Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions from your PDF using AI-powered RAG'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = os.path.join(
    "data",
    "uploads",
)

CHROMA_ROOT = "chroma_db"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)

os.makedirs(
    CHROMA_ROOT,
    exist_ok=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    st.divider()

    # Temporary retrieval-only test mode.
    # When enabled, Gemini generation is skipped so we can verify
    # whether the correct PDF chunks are being retrieved.
    debug_retrieval = st.checkbox(
        "🔍 Retrieval Test Mode",
        value=False,
        help="Shows the top retrieved chunks without calling Gemini.",
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# SELECT ACTIVE PDF
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Uploaded PDF
    # --------------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    # Create hash from ACTUAL PDF content
    document_key = hashlib.md5(
        file_bytes
    ).hexdigest()[:16]

    document_name = os.path.basename(
        uploaded_file.name
    )

    pdf_path = os.path.join(
        UPLOAD_DIR,
        f"{document_key}_{document_name}",
    )

    # Save uploaded PDF
    if not os.path.exists(pdf_path):

        with open(
            pdf_path,
            "wb",
        ) as f:

            f.write(file_bytes)

else:

    # --------------------------------------------------------
    # Default PDF
    # --------------------------------------------------------

    pdf_path = os.path.join(
        "data",
        "AI_ML_Guide.pdf",
    )

    if not os.path.exists(pdf_path):

        st.error(
            "No PDF uploaded and "
            "AI_ML_Guide.pdf was not found."
        )

        st.stop()

    with open(
        pdf_path,
        "rb",
    ) as f:

        file_bytes = f.read()

    # Hash actual file content
    document_key = hashlib.md5(
        file_bytes
    ).hexdigest()[:16]

    document_name = os.path.basename(
        pdf_path
    )


# ============================================================
# RESET CHAT WHEN PDF CHANGES
# ============================================================

if (
    st.session_state.active_document_key
    != document_key
):

    st.session_state.active_document_key = (
        document_key
    )

    st.session_state.messages = []


# ============================================================
# PDF-SPECIFIC CHROMA STORAGE
# ============================================================

document_chroma_dir = os.path.join(
    CHROMA_ROOT,
    document_key,
)

collection_name = (
    f"pdf_{document_key}"
)


# ============================================================
# LOAD PDF
# ============================================================

@st.cache_data
def load_pdf(
    pdf_path,
    document_key,
):

    documents = []

    pdf = fitz.open(
        pdf_path
    )

    for page_number, page in enumerate(pdf):

        text = page.get_text(
            "text"
        ).strip()

        if text:

            documents.append(
                Document(
                    page_content=text,

                    metadata={
                        "source": pdf_path,
                        "page": page_number + 1,
                        "document_id": document_key,
                    },
                )
            )

    pdf.close()

    return documents


# ============================================================
# CREATE CHUNKS
# ============================================================

@st.cache_data
def create_chunks(
    pdf_path,
    document_key,
):

    documents = load_pdf(
        pdf_path,
        document_key,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def create_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )


# ============================================================
# VECTOR STORE
# ============================================================

@st.cache_resource
def create_vector_store(
    pdf_path,
    document_key,
    collection_name,
    document_chroma_dir,
):

    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=document_chroma_dir,
    )

    chunks = create_chunks(
        pdf_path,
        document_key,
    )

    # --------------------------------------------------------
    # Unique IDs for this exact PDF
    # --------------------------------------------------------

    document_ids = []

    for i, chunk in enumerate(chunks):

        page = chunk.metadata.get(
            "page",
            0,
        )

        chunk_id = (
            f"{document_key}"
            f"-page-{page}"
            f"-chunk-{i}"
        )

        document_ids.append(
            chunk_id
        )

    # --------------------------------------------------------
    # Existing IDs
    # --------------------------------------------------------

    existing_data = vector_store.get()

    existing_ids = set(
        existing_data.get(
            "ids",
            [],
        )
    )

    # --------------------------------------------------------
    # Add only new chunks
    # --------------------------------------------------------

    new_chunks = []
    new_ids = []

    for chunk, chunk_id in zip(
        chunks,
        document_ids,
    ):

        if chunk_id not in existing_ids:

            new_chunks.append(
                chunk
            )

            new_ids.append(
                chunk_id
            )

    if new_chunks:

        vector_store.add_documents(
            documents=new_chunks,
            ids=new_ids,
        )

    return vector_store


# ============================================================
# GEMINI LLM
# ============================================================

@st.cache_resource
def create_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        timeout=120,
        max_retries=1,
    )


# ============================================================
# LOAD CURRENT DOCUMENT
# ============================================================

documents = load_pdf(
    pdf_path,
    document_key,
)


# ============================================================
# CHECK PDF TEXT
# ============================================================

if not documents:

    st.error(
        "No readable text was found in this PDF. "
        "The PDF may be scanned or image-based."
    )

    st.stop()


# ============================================================
# CREATE CHUNKS
# ============================================================

chunks = create_chunks(
    pdf_path,
    document_key,
)


# ============================================================
# CREATE VECTOR STORE
# ============================================================

vector_store = create_vector_store(
    pdf_path,
    document_key,
    collection_name,
    document_chroma_dir,
)


# ============================================================
# CREATE LLM
# ============================================================

llm = create_llm()


# ============================================================
# SIDEBAR DOCUMENT INFO
# ============================================================

with st.sidebar:

    st.success(
        "PDF Ready"
    )

    st.write(
        f"**File:** {document_name}"
    )

    st.write(
        f"**Pages:** {len(documents)}"
    )

    st.write(
        f"**Chunks:** {len(chunks)}"
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask something about your PDF..."
)


if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            query
        )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )


    # --------------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Searching your PDF..."
    ):

        try:

            retrieved_docs = (
                vector_store.similarity_search(
                    query,
                    k=3,
                )
            )

        except Exception as e:

            st.error(
                "Unable to search the PDF."
            )

            st.caption(
                f"Technical error: {str(e)}"
            )

            st.stop()


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    valid_docs = []

    for doc in retrieved_docs:

        doc_id = doc.metadata.get(
            "document_id"
        )

        if doc_id == document_key:

            valid_docs.append(
                doc
            )


    retrieved_docs = valid_docs


    # --------------------------------------------------------
    # RETRIEVAL TEST MODE
    # --------------------------------------------------------

    if debug_retrieval:

        st.success(
            f"Retrieved {len(retrieved_docs)} valid chunk(s) "
            f"from: {document_name}"
        )

        if not retrieved_docs:

            st.warning(
                "No chunks were retrieved from the active PDF."
            )

        else:

            st.markdown("### 🔍 Top Retrieved Chunks")

            for i, doc in enumerate(retrieved_docs, 1):

                page = doc.metadata.get(
                    "page",
                    "Unknown",
                )

                st.markdown(
                    f"#### Result {i} — 📄 Page {page}"
                )

                st.code(
                    doc.page_content[:1500],
                    language=None,
                )

                st.caption(
                    f"Document ID: "
                    f"{doc.metadata.get('document_id', 'Unknown')}"
                )

                st.divider()

        st.info(
            "Gemini generation was skipped. "
            "This test only checks PDF retrieval."
        )

        st.stop()


    # --------------------------------------------------------
    # NO RETRIEVED DOCUMENT
    # --------------------------------------------------------

    if not retrieved_docs:

        answer = (
            "I could not find the answer "
            "in the provided document."
        )

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.stop()


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for doc in retrieved_docs:

        page = doc.metadata.get(
            "page",
            "Unknown",
        )

        context_parts.append(
            f"Page {page}:\n"
            f"{doc.page_content}"
        )


    context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # RAG PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the context
provided below.

Do not use outside knowledge.

If the answer is not present in the context, say:

"I could not find the answer in the provided document."

Keep the answer clear and easy to understand.

Context:
{context}

Question:
{query}

Answer:
"""


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        try:

            with st.spinner(
                "🤖 Generating answer..."
            ):

                response = llm.invoke(
                    prompt
                )


            # ------------------------------------------------
            # HANDLE GEMINI 3.x RESPONSE
            # ------------------------------------------------

            if isinstance(
                response.content,
                list,
            ):

                answer_parts = []

                for item in response.content:

                    if isinstance(
                        item,
                        dict,
                    ):

                        text = item.get(
                            "text",
                            "",
                        )

                        if text:

                            answer_parts.append(
                                text
                            )

                answer = "\n".join(
                    answer_parts
                ).strip()

            else:

                answer = str(
                    response.content
                ).strip()


            if not answer:

                answer = (
                    "I could not generate "
                    "an answer."
                )


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            st.markdown(
                answer
            )


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            with st.expander(
                "📖 View Retrieved Sources"
            ):

                for doc in retrieved_docs:

                    page = doc.metadata.get(
                        "page",
                        "Unknown",
                    )

                    snippet = (
                        doc.page_content[:500]
                    )

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-title">
                                📄 Page {page}
                            </div>

                            <div class="source-text">
                                {html.escape(snippet)}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


            # ------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


        # ====================================================
        # ERROR HANDLING
        # ====================================================

        except Exception as e:

            error_text = str(e)

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED"
                in error_text
            ):

                st.warning(
                    "⏳ Gemini API free-tier quota "
                    "is currently exhausted. "
                    "Please wait until the quota "
                    "resets before trying again."
                )

            else:

                st.error(
                    "Sorry, something went wrong "
                    "while generating the answer."
                )

                st.caption(
                    f"Technical error: {error_text}"
                )