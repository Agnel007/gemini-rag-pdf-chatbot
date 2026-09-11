import pymupdf
import streamlit as st

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI / ML PDF Assistant",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# 2. Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #777;
        margin-bottom: 2rem;
    }

    .info-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    .source-card {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 3. Load Environment Variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 4. Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 AI / ML PDF Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions and get answers directly from your PDF.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# 5. Load PDF
# --------------------------------------------------

@st.cache_resource
def load_pdf():

    pdf_path = "data/AI_ML_Guide.pdf"

    pdf = pymupdf.open(pdf_path)

    documents = []

    for page_number, page in enumerate(pdf):

        text = page.get_text()

        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_number + 1
                    }
                )
            )

    pdf.close()

    return documents


# --------------------------------------------------
# 6. Create Chunks
# --------------------------------------------------

@st.cache_resource
def create_chunks():

    documents = load_pdf()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)


# --------------------------------------------------
# 7. Create / Load ChromaDB
# --------------------------------------------------

@st.cache_resource
def create_vector_store():

    chunks = create_chunks()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_store = Chroma(
        collection_name="ai_ml_guide",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

    document_ids = [
        f"page-{chunk.metadata['page']}-chunk-{i}"
        for i, chunk in enumerate(chunks)
    ]

    existing_ids = set(
        vector_store.get()["ids"]
    )

    new_chunks = []
    new_ids = []

    for chunk, document_id in zip(
        chunks,
        document_ids
    ):

        if document_id not in existing_ids:

            new_chunks.append(chunk)
            new_ids.append(document_id)

    if new_chunks:

        vector_store.add_documents(
            documents=new_chunks,
            ids=new_ids
        )

    return vector_store


# --------------------------------------------------
# 8. Create Gemini LLM
# --------------------------------------------------

@st.cache_resource
def create_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        timeout=60,
        max_retries=1
    )


# --------------------------------------------------
# 9. Load Resources
# --------------------------------------------------

with st.spinner("Loading AI / ML knowledge base..."):

    documents = load_pdf()
    chunks = create_chunks()
    vector_store = create_vector_store()
    llm = create_llm()


# --------------------------------------------------
# 10. Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.markdown(
        f"""
        <div class="info-card">
        <b>Document</b><br>
        AI / ML Guide
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric(
        "Pages",
        len(documents)
    )

    st.metric(
        "Chunks",
        len(chunks)
    )

    st.divider()

    st.write("### ⚙️ Configuration")

    st.write("**LLM:** Gemini 3.6 Flash")

    st.write("**Embeddings:** Gemini Embedding")

    st.write("**Vector DB:** ChromaDB")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# 11. Chat History
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# 12. Display Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# 13. Chat Input
# --------------------------------------------------

query = st.chat_input(
    "Ask something about the AI / ML document..."
)


if query:

    # --------------------------------------------------
    # User Message
    # --------------------------------------------------

    with st.chat_message("user"):

        st.markdown(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    with st.spinner(
        "🔎 Searching the document..."
    ):

        results = vector_store.similarity_search(
            query,
            k=3
        )


    # --------------------------------------------------
    # Create Context
    # --------------------------------------------------

    context = "\n\n".join(
        result.page_content
        for result in results
    )


    # --------------------------------------------------
    # RAG Prompt
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Generating answer..."
        ):

            try:

                response = llm.invoke(prompt)

                if isinstance(
                    response.content,
                    list
                ):

                    answer = ""

                    for item in response.content:

                        if (
                            isinstance(item, dict)
                            and item.get("type") == "text"
                        ):

                            answer += item.get(
                                "text",
                                ""
                            )

                else:

                    answer = str(
                        response.content
                    )

                st.markdown(answer)

            except Exception as e:

                answer = (
                    "Sorry, something went wrong "
                    "while generating the answer."
                )

                st.error(str(e))

                st.markdown(answer)


    # --------------------------------------------------
    # Save Assistant Message
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # --------------------------------------------------
    # Sources
    # --------------------------------------------------

    with st.expander(
        "📖 View Retrieved Sources"
    ):

        seen_pages = set()

        for i, result in enumerate(results):

            page_number = result.metadata["page"]

            if page_number not in seen_pages:

                seen_pages.add(page_number)

                st.markdown(
                    f"**Source {i + 1} — Page {page_number}**"
                )

                st.caption(
                    result.page_content[:400]
                    + "..."
                )

                st.divider()