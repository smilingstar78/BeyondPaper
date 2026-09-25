import os
import requests
import hashlib
import chromadb

from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# CHROMA - IN MEMORY
# =========================================================

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="research_papers"
)


# =========================================================
# EMBEDDINGS
# =========================================================

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

EMBEDDING_URL = os.getenv("EMBEDDING_URL")


def create_embeddings(texts):

    response = requests.post(
        EMBEDDING_URL,
        headers={
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "input": texts
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    return [
        item["embedding"]
        for item in data["data"]
    ]


# =========================================================
# DOWNLOAD PDF
# =========================================================

def download_pdf(pdf_url):

    print(f"Downloading PDF: {pdf_url}")

    response = requests.get(
        pdf_url,
        timeout=60,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    if not response.content.startswith(b"%PDF"):

        content_type = response.headers.get(
            "content-type",
            ""
        )

        raise ValueError(
            f"URL did not return a PDF. "
            f"Content-Type: {content_type}"
        )

    return BytesIO(response.content)


# =========================================================
# READ PDF
# =========================================================

def read_pdf(pdf_url):

    pdf_file = download_pdf(pdf_url)

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            page_text = page.extract_text()

            if page_text:

                pages.append({
                    "page": page_number,
                    "text": page_text
                })

        except Exception as e:

            print(
                f"Could not read page "
                f"{page_number}: {e}"
            )

    if not pages:

        raise ValueError(
            "PDF contains no extractable text."
        )

    return pages


# =========================================================
# CHUNK TEXT
# =========================================================

def chunk_text(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    all_chunks = []

    for page in pages:

        chunks = splitter.split_text(
            page["text"]
        )

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return all_chunks


# =========================================================
# STORE PAPER
# =========================================================

def store_paper(title, pdf_url):

    print(f"\nReading: {title}")

    pages = read_pdf(pdf_url)

    total_characters = sum(
        len(page["text"])
        for page in pages
    )

    print(
        f"Extracted "
        f"{total_characters} characters."
    )

    chunks = chunk_text(pages)

    print(
        f"Created {len(chunks)} chunks."
    )

    if not chunks:

        raise ValueError(
            "No chunks were created."
        )

    paper_id = hashlib.md5(
        pdf_url.encode()
    ).hexdigest()

    ids = [
        f"{paper_id}_{i}"
        for i in range(len(chunks))
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "Creating embeddings..."
    )

    vectors = create_embeddings(
        documents
    )

    collection.upsert(

        ids=ids,

        documents=documents,

        embeddings=vectors,

        metadatas=[

            {
                "title": title,
                "pdf_url": pdf_url,
                "page": chunk["page"],
                "chunk_index": i
            }

            for i, chunk in enumerate(chunks)
        ]
    )

    print(
        f"Stored {len(chunks)} "
        f"chunks in ChromaDB."
    )


# =========================================================
# SEARCH PAPER
# =========================================================

def search_paper(
    question,
    title,
    k=2
):

    query_embedding = create_embeddings(
        [question]
    )[0]

    try:

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=k,

            where={
                "title": title
            }
        )

    except Exception as e:

        print(
            f"Chroma search error: {e}"
        )

        return []

    documents = results.get(
        "documents"
    )

    metadatas = results.get(
        "metadatas"
    )

    if not documents:

        return []

    documents = documents[0]

    metadatas = (
        metadatas[0]
        if metadatas
        else [{} for _ in documents]
    )

    retrieved_chunks = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        page = metadata.get(
            "page",
            "unknown"
        )

        retrieved_chunks.append(
            f"[Page {page}]\n{document}"
        )

    return retrieved_chunks
