import os
import pymupdf

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Load PDF
# --------------------------------------------------

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

print(f"Total pages loaded: {len(documents)}")


# --------------------------------------------------
# 3. Split PDF into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# --------------------------------------------------
# 4. Create Gemini Embeddings
# --------------------------------------------------

print("\nCreating embeddings...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

print("Embeddings model ready.")


# --------------------------------------------------
# 5. Create / Load ChromaDB
# --------------------------------------------------

print("\nCreating ChromaDB...")

vector_store = Chroma(
    collection_name="ai_ml_guide",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

print("ChromaDB ready.")


# --------------------------------------------------
# 6. Store chunks in ChromaDB
# --------------------------------------------------

# Use unique IDs so the same chunks are not duplicated

document_ids = [
    f"page-{chunk.metadata['page']}-chunk-{i}"
    for i, chunk in enumerate(chunks)
]

existing_ids = set(
    vector_store.get()["ids"]
)

new_chunks = []
new_ids = []

for chunk, document_id in zip(chunks, document_ids):

    if document_id not in existing_ids:
        new_chunks.append(chunk)
        new_ids.append(document_id)


if new_chunks:

    vector_store.add_documents(
        documents=new_chunks,
        ids=new_ids
    )


print(f"New chunks added: {len(new_chunks)}")
print(f"Total chunks available: {len(chunks)}")


# --------------------------------------------------
# 7. Create Gemini LLM
# --------------------------------------------------

print("\nLoading Gemini LLM...")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    timeout=60,
    max_retries=1
)

print("LLM ready.")


# --------------------------------------------------
# 8. Interactive RAG Chatbot
# --------------------------------------------------

print("\n======================================")
print("       AI / ML RAG CHATBOT")
print("======================================")
print("Ask questions about the PDF.")
print("Type 'exit' or 'quit' to stop.")
print("======================================")


while True:

    # Get user question

    query = input("\nYou: ").strip()


    # Exit condition

    if query.lower() in ["exit", "quit"]:

        print("\nBot: Goodbye! 👋")
        break


    # Prevent empty questions

    if not query:

        print("Bot: Please enter a question.")
        continue


    # --------------------------------------------------
    # 9. Semantic Search / Retrieval
    # --------------------------------------------------

    print("\nSearching the document...")

    results = vector_store.similarity_search(
        query,
        k=3
    )


    # --------------------------------------------------
    # 10. Display Retrieved Sources
    # --------------------------------------------------

    print("\n--- Retrieved Sources ---")

    for i, result in enumerate(results):

        print(
            f"Source {i + 1}: "
            f"Page {result.metadata['page']}"
        )


    # --------------------------------------------------
    # 11. Combine Retrieved Chunks
    # --------------------------------------------------

    context = "\n\n".join(
        result.page_content
        for result in results
    )


    # --------------------------------------------------
    # 12. Create RAG Prompt
    # --------------------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the context provided below.

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
    # 13. Generate Final Answer
    # --------------------------------------------------

    print("\nBot: Generating answer...")

    try:

        response = llm.invoke(prompt)

    except Exception as e:

        print("\nBot: Gemini API error occurred.")
        print("Error:", e)
        continue


    # --------------------------------------------------
    # 14. Extract Clean Text
    # --------------------------------------------------

    if isinstance(response.content, list):

        answer = ""

        for item in response.content:

            if isinstance(item, dict) and item.get("type") == "text":

                answer += item.get("text", "")

    else:

        answer = str(response.content)


    print("\nBot:", answer)