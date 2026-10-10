
import os
import hashlib
import logging
from io import BytesIO

import requests
import chromadb

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# ENVIRONMENT AND LOGGING
# =========================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_EMBEDDING_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-embedding-001:batchEmbedContents"
)

GEMINI_QUERY_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-embedding-001:embedContent"
)

logger.info(
    "GEMINI_API_KEY configured: %s",
    bool(GEMINI_API_KEY)
)


# =========================================================
# CHROMA — IN MEMORY
# =========================================================

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="research_papers"
)


# =========================================================
# GEMINI DOCUMENT EMBEDDINGS
# =========================================================

def create_document_embeddings(texts, title):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from the environment."
        )

    embeddings = []
    batch_size = 20

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        requests_body = [
            {
                "model": "models/gemini-embedding-001",
                "content": {
                    "parts": [
                        {
                            "text": f"title: {title} | text: {text}"
                        }
                    ]
                },
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": 768,
            }
            for text in batch
        ]

        try:
            response = requests.post(
                GEMINI_EMBEDDING_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GEMINI_API_KEY,
                },
                json={"requests": requests_body},
                timeout=60,
            )

            if not response.ok:
                logger.error(
                    "Gemini document embedding failed. "
                    "HTTP %s: %s",
                    response.status_code,
                    response.text[:1500],
                )

            response.raise_for_status()
            data = response.json()

            batch_embeddings = data.get("embeddings", [])

            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    "Gemini returned an unexpected number "
                    "of document embeddings."
                )

            for item in batch_embeddings:
                values = item.get("values")

                if not values:
                    raise ValueError(
                        "Gemini returned an empty document embedding."
                    )

                embeddings.append(values)

        except Exception:
            logger.exception(
                "Document embedding batch failed. "
                "Batch starts at %s for paper '%s'.",
                start,
                title,
            )
            raise

    return embeddings


# =========================================================
# GEMINI QUERY EMBEDDING
# =========================================================

def create_query_embedding(question):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from the environment."
        )

    try:
        response = requests.post(
            GEMINI_QUERY_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY,
            },
            json={
                "model": "models/gemini-embedding-001",
                "content": {
                    "parts": [{"text": question}]
                },
                "taskType": "RETRIEVAL_QUERY",
                "outputDimensionality": 768,
            },
            timeout=30,
        )

        if not response.ok:
            logger.error(
                "Gemini query embedding failed. HTTP %s: %s",
                response.status_code,
                response.text[:1500],
            )

        response.raise_for_status()

        data = response.json()
        embedding = data.get("embedding", {})
        values = embedding.get("values")

        if not values:
            raise ValueError(
                "Gemini did not return a valid query embedding."
            )

        return values

    except Exception:
        logger.exception("Query embedding creation failed.")
        raise


# =========================================================
# DOWNLOAD PDF
# =========================================================

def download_pdf(pdf_url):
    if not pdf_url:
        raise ValueError("The paper has no PDF URL.")

    logger.info("Downloading PDF: %s", pdf_url)

    try:
        response = requests.get(
            pdf_url,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0 BeyondPaper/1.0"},
            allow_redirects=True,
        )

        if not response.ok:
            logger.error(
                "PDF download failed. HTTP %s for %s",
                response.status_code,
                pdf_url,
            )

        response.raise_for_status()

        if not response.content.startswith(b"%PDF"):
            content_type = response.headers.get(
                "content-type", "unknown"
            )

            raise ValueError(
                "The supplied URL did not return a PDF. "
                f"Content-Type: {content_type}; "
                f"first bytes: {response.content[:30]!r}"
            )

        return BytesIO(response.content)

    except Exception:
        logger.exception("PDF download failed: %s", pdf_url)
        raise


# =========================================================
# READ PDF
# =========================================================

def read_pdf(pdf_url):
    pdf_file = download_pdf(pdf_url)
    reader = PdfReader(pdf_file)

    if reader.is_encrypted:
        try:
            result = reader.decrypt("")
            if result == 0:
                raise ValueError("The PDF is password-protected.")
        except Exception as exc:
            raise ValueError(
                f"Could not decrypt the PDF: {exc}"
            ) from exc

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()

            if page_text and page_text.strip():
                pages.append(
                    {
                        "page": page_number,
                        "text": page_text,
                    }
                )

        except Exception:
            logger.exception(
                "Could not extract text from PDF page %s.",
                page_number,
            )

    if not pages:
        raise ValueError(
            "The PDF contains no extractable text. "
            "It may be scanned, image-only, or unsupported."
        )

    logger.info(
        "Extracted text from %s pages.",
        len(pages),
    )

    return pages


# =========================================================
# CHUNK TEXT
# =========================================================

def chunk_text(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    all_chunks = []

    for page in pages:
        chunks = splitter.split_text(page["text"])

        for chunk in chunks:
            all_chunks.append(
                {
                    "text": chunk,
                    "page": page["page"],
                }
            )

    return all_chunks


# =========================================================
# STORE PAPER
# =========================================================

def store_paper(title, pdf_url):
    logger.info("Starting paper processing: %s", title)

    try:
        pages = read_pdf(pdf_url)

        total_characters = sum(
            len(page["text"]) for page in pages
        )

        logger.info(
            "Extracted %s characters from '%s'.",
            total_characters,
            title,
        )

        chunks = chunk_text(pages)

        if not chunks:
            raise ValueError(
                f"No text chunks were created for '{title}'."
            )

        logger.info(
            "Created %s chunks for '%s'.",
            len(chunks),
            title,
        )

        paper_id = hashlib.sha256(
            pdf_url.encode("utf-8")
        ).hexdigest()[:32]

        ids = [
            f"{paper_id}_{i}"
            for i in range(len(chunks))
        ]

        documents = [
            chunk["text"] for chunk in chunks
        ]

        vectors = create_document_embeddings(
            documents,
            title,
        )

        if len(vectors) != len(documents):
            raise ValueError(
                "The number of embeddings does not match "
                "the number of text chunks."
            )

        if any(
            len(vector) != len(vectors[0])
            for vector in vectors
        ):
            raise ValueError(
                "Gemini returned embeddings with inconsistent dimensions."
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
                    "chunk_index": i,
                }
                for i, chunk in enumerate(chunks)
            ],
        )

        logger.info(
            "Successfully indexed %s chunks for '%s'.",
            len(chunks),
            title,
        )

        return True

    except Exception as exc:
        logger.exception(
            "Failed to read/index paper '%s'. URL: %s",
            title,
            pdf_url,
        )

        # IMPORTANT:
        # Pass the actual failure to main.py instead of
        # hiding it behind a False return value.
        raise RuntimeError(
            f"Failed to process '{title}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# =========================================================
# SEARCH PAPER CHUNKS
# =========================================================

DEFAULT_QUERIES = [
    "research problem and objective",
    "method or approach used",
    "limitations and weaknesses",
    "future work and open questions",
]


def _query_chunks(question, title, k):
    try:
        query_embedding = create_query_embedding(question)

        count = collection.count()

        if count == 0:
            logger.warning("Chroma collection is empty.")
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            where={"title": title},
        )

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        if not documents or not documents[0]:
            return []

        docs = documents[0]

        metadata_list = (
            metadatas[0]
            if metadatas and metadatas[0]
            else [{} for _ in docs]
        )

        return [
            (
                metadata.get("page", "unknown"),
                document,
            )
            for document, metadata in zip(
                docs,
                metadata_list,
            )
        ]

    except Exception:
        logger.exception(
            "Chroma search failed for paper '%s', query '%s'.",
            title,
            question,
        )
        raise


def search_paper(title, questions=None, k=2):
    queries = questions or DEFAULT_QUERIES

    seen = set()
    retrieved_chunks = []

    for question in queries:
        results = _query_chunks(
            question,
            title,
            k,
        )

        for page, document in results:
            key = (page, document)

            if key in seen:
                continue

            seen.add(key)

            retrieved_chunks.append(
                f"[Page {page}]\n{document}"
            )

    if not retrieved_chunks:
        logger.warning(
            "No chunks retrieved for paper '%s'.",
            title,
        )

    return retrieved_chunks
