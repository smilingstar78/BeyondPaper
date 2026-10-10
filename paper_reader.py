
import os
import time
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

EMBEDDING_DIMENSIONS = 768
EMBEDDING_BATCH_SIZE = 20
MAX_RETRIES = 3

logger.info(
    "GEMINI_API_KEY configured: %s",
    bool(GEMINI_API_KEY),
)


# =========================================================
# CHROMA — IN MEMORY
# =========================================================

# Suitable for indexing and querying within the same invocation.
# Do not rely on this collection persisting between Vercel requests.

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="research_papers"
)


# =========================================================
# GEMINI REQUEST WITH LIMITED RETRIES
# =========================================================

def gemini_post(url, payload, timeout=60):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from the Vercel environment."
        )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.ok:
                return response

            error_text = response.text[:2000]

            # Do not keep retrying if the response clearly indicates
            # that the account's quota has been exhausted.
            lower_error = error_text.lower()

            permanent_quota_error = (
                response.status_code == 429
                and any(
                    phrase in lower_error
                    for phrase in (
                        "quota exceeded",
                        "per day",
                        "daily limit",
                    )
                )
            )

            if permanent_quota_error:
                raise RuntimeError(
                    "Gemini API quota appears to be exhausted. "
                    "Check Gemini API usage and limits. "
                    f"Response: {error_text}"
                )

            if (
                response.status_code not in (429, 500, 502, 503, 504)
                or attempt == MAX_RETRIES - 1
            ):
                raise RuntimeError(
                    f"Gemini API returned HTTP "
                    f"{response.status_code}: {error_text}"
                )

            # Retry temporary rate limits and server errors.
            retry_after = response.headers.get("Retry-After")

            try:
                wait_seconds = float(retry_after)
            except (TypeError, ValueError):
                wait_seconds = 2 ** (attempt + 1)

            wait_seconds = min(max(wait_seconds, 1), 8)

            logger.warning(
                "Gemini returned HTTP %s. Retry %s/%s in %s seconds.",
                response.status_code,
                attempt + 1,
                MAX_RETRIES - 1,
                wait_seconds,
            )

            time.sleep(wait_seconds)

        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Could not reach Gemini API: {exc}"
                ) from exc

            wait_seconds = 2 ** (attempt + 1)

            logger.warning(
                "Gemini network error. Retry %s/%s in %s seconds: %s",
                attempt + 1,
                MAX_RETRIES - 1,
                wait_seconds,
                exc,
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Gemini request failed after retries.")


# =========================================================
# GEMINI DOCUMENT EMBEDDINGS
# =========================================================

def create_document_embeddings(texts, title):
    if not texts:
        return []

    embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]

        request_items = [
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
                "outputDimensionality": EMBEDDING_DIMENSIONS,
            }
            for text in batch
        ]

        logger.info(
            "Embedding batch %s for '%s' (%s chunks).",
            start // EMBEDDING_BATCH_SIZE + 1,
            title,
            len(batch),
        )

        response = gemini_post(
            GEMINI_EMBEDDING_URL,
            {"requests": request_items},
            timeout=60,
        )

        data = response.json()
        batch_embeddings = data.get("embeddings", [])

        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "Gemini returned an unexpected number of embeddings "
                f"for '{title}'. Expected {len(batch)}, "
                f"received {len(batch_embeddings)}."
            )

        for item in batch_embeddings:
            values = item.get("values")

            if not values:
                raise RuntimeError(
                    f"Gemini returned an empty embedding for '{title}'."
                )

            if len(values) != EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    "Unexpected embedding dimensions: "
                    f"{len(values)} instead of {EMBEDDING_DIMENSIONS}."
                )

            embeddings.append(values)

    return embeddings


# =========================================================
# GEMINI QUERY EMBEDDING
# =========================================================

def create_query_embedding(question):
    response = gemini_post(
        GEMINI_QUERY_URL,
        {
            "model": "models/gemini-embedding-001",
            "content": {
                "parts": [{"text": question}]
            },
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": EMBEDDING_DIMENSIONS,
        },
        timeout=30,
    )

    data = response.json()
    values = data.get("embedding", {}).get("values")

    if not values:
        raise RuntimeError(
            "Gemini did not return a valid query embedding."
        )

    if len(values) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            "Unexpected query embedding dimensions: "
            f"{len(values)} instead of {EMBEDDING_DIMENSIONS}."
        )

    return values


# =========================================================
# DOWNLOAD PDF
# =========================================================

def download_pdf(pdf_url):
    if not pdf_url:
        raise ValueError("The paper has no PDF URL.")

    logger.info("Downloading PDF: %s", pdf_url)

    response = requests.get(
        pdf_url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 BeyondPaper/1.0"},
        allow_redirects=True,
    )
    response.raise_for_status()

    if not response.content.startswith(b"%PDF"):
        raise ValueError(
            "The supplied URL did not return a PDF. "
            f"Content-Type: {response.headers.get('content-type', 'unknown')}; "
            f"first bytes: {response.content[:30]!r}"
        )

    return BytesIO(response.content)


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
                pages.append({
                    "page": page_number,
                    "text": page_text,
                })

        except Exception:
            logger.exception(
                "Could not extract text from PDF page %s.",
                page_number,
            )

    if not pages:
        raise ValueError(
            "The PDF contains no extractable text. "
            "It may be scanned or image-only."
        )

    logger.info("Extracted text from %s pages.", len(pages))
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
        for chunk in splitter.split_text(page["text"]):
            all_chunks.append({
                "text": chunk,
                "page": page["page"],
            })

    return all_chunks


# =========================================================
# STORE PAPER
# =========================================================

def store_paper(title, pdf_url):
    logger.info("Starting paper processing: %s", title)

    try:
        pages = read_pdf(pdf_url)
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

        documents = [chunk["text"] for chunk in chunks]

        vectors = create_document_embeddings(documents, title)

        if len(vectors) != len(documents):
            raise RuntimeError(
                "The number of embeddings does not match "
                "the number of text chunks."
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
            "Failed to process paper '%s'. URL: %s",
            title,
            pdf_url,
        )

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
    query_embedding = create_query_embedding(question)

    # Count only this paper's chunks. Counting the whole collection
    # can request more results than exist for this particular title.
    count = collection.count(
        where={"title": title}
    )

    if count == 0:
        logger.warning("No indexed chunks found for '%s'.", title)
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, count),
        where={"title": title},
        include=["documents", "metadatas"],
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
        (metadata.get("page", "unknown"), document)
        for document, metadata in zip(docs, metadata_list)
    ]


def search_paper(question, title=None, k=2, questions=None):
    """
    Supports both calling styles:

    Current local main.py:
        search_paper(question, title, k=2)

    Older Vercel style:
        search_paper(title, questions=[...], k=2)
    """

    if title is None:
        # Backwards compatibility with the older title-first interface.
        title = question
        queries = questions or DEFAULT_QUERIES
    else:
        # Local main.py passes one question followed by the paper title.
        queries = question if isinstance(question, list) else [question]

    seen = set()
    retrieved_chunks = []

    for query in queries:
        results = _query_chunks(query, title, k)

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
