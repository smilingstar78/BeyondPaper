import os
import hashlib
import requests
import chromadb

from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# ENVIRONMENT
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_EMBEDDING_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-embedding-001:batchEmbedContents"
)


# =========================================================
# CHROMA - IN MEMORY
# =========================================================

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="research_papers"
)


# =========================================================
# GEMINI EMBEDDINGS
# =========================================================

def create_document_embeddings(texts, title):

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured."
        )

    embeddings = []

    # Gemini API request sizes should stay reasonable.
    batch_size = 50

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        requests_body = []

        for text in batch:

            requests_body.append(
                {
                    "model": "models/gemini-embedding-001",
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    f"title: {title} | "
                                    f"text: {text}"
                                )
                            }
                        ]
                    },
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "outputDimensionality": 768
                }
            )

        response = requests.post(
            GEMINI_EMBEDDING_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            },
            json={
                "requests": requests_body
            },
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        batch_embeddings = data.get(
            "embeddings",
            []
        )

        if len(batch_embeddings) != len(batch):

            raise ValueError(
                "Gemini returned an unexpected "
                "number of embeddings."
            )

        embeddings.extend(
            [
                item["values"]
                for item in batch_embeddings
            ]
        )

    return embeddings


def create_query_embedding(question):

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured."
        )

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-embedding-001:embedContent"
    )

    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        },
        json={
            "model": "models/gemini-embedding-001",
            "content": {
                "parts": [
                    {
                        "text": question
                    }
                ]
            },
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": 768
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    embedding = data.get("embedding")

    if not embedding:
        raise ValueError(
            "Gemini did not return a query embedding."
        )

    return embedding["values"]


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

                pages.append(
                    {
                        "page": page_number,
                        "text": page_text
                    }
                )

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

            all_chunks.append(
                {
                    "text": chunk,
                    "page": page["page"]
                }
            )

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
        "Creating Gemini embeddings..."
    )

    vectors = create_document_embeddings(
        documents,
        title
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
        f"Stored {len(chunks)} chunks."
    )


# =========================================================
# SEARCH PAPER
# =========================================================

def search_paper(
    question,
    title,
    k=2
):

    try:

        query_embedding = (
            create_query_embedding(
                question
            )
        )

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
