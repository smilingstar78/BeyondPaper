import os
import re
from io import BytesIO

import feedparser
import requests
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.tools import tool


load_dotenv()

# -----------------------------
# Settings
# -----------------------------

REQUEST_TIMEOUT = 30
MAX_PDF_BYTES = 25 * 1024 * 1024      # refuse PDFs bigger than 25 MB
MAX_PAPER_CHARS = 7000                # text returned to the LLM per paper
MAX_ABSTRACT_CHARS = 900              # abstract length returned per search hit
HEADERS = {"User-Agent": "research-agent/1.0 (educational project)"}


# -----------------------------
# Helpers
# -----------------------------

def _rebuild_abstract(inverted_index):
    """
    OpenAlex stores abstracts as an inverted index:
        {"word": [position1, position2], ...}
    Turn it back into normal text so the LLM can actually read it.
    """
    if not inverted_index:
        return ""

    positions = []

    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))

    positions.sort(key=lambda x: x[0])

    return " ".join(word for _, word in positions)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _shorten(text: str, limit: int) -> str:
    text = _clean_text(text or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _to_pdf_url(url: str) -> str:
    """Convert an arXiv abstract link into its PDF link."""
    url = url.strip()

    if "arxiv.org/abs/" in url:
        url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/")

    return url


def _select_paper_text(text: str, max_chars: int) -> str:
    """
    Keep the parts of a paper that matter most for research gaps:
    the start (abstract + introduction) and the end
    (conclusion, limitations, future work) - without the reference list.
    """
    text = _clean_text(text)

    # Cut off the reference list if it is in the second half of the paper
    cut = max(text.rfind("References"), text.rfind("REFERENCES"))

    if cut > len(text) * 0.5:
        text = text[:cut]

    if len(text) <= max_chars:
        return text

    head_len = int(max_chars * 0.55)
    tail_len = max_chars - head_len

    return (
        text[:head_len]
        + "\n\n[... middle of paper omitted ...]\n\n"
        + text[-tail_len:]
    )


# -----------------------------
# Tool 1: OpenAlex
# -----------------------------

@tool
def search_open_alex(query: str):
    """
    Search OpenAlex for research papers related to a topic.
    Returns title, year, citation count, DOI, abstract and
    an open-access PDF link (when one exists).
    """

    url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "per-page": 5,
    }

    headers = dict(HEADERS)

    # OpenAlex now requires a (free) API key: https://openalex.org/settings/api
    api_key = os.getenv("OPENALEX_API_KEY")

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        print("\n[OpenAlex] status code:", response.status_code)

        if response.status_code in (401, 403, 409, 429):
            return {
                "error": (
                    f"OpenAlex returned HTTP {response.status_code}. "
                    "An API key (OPENALEX_API_KEY) may be missing or the "
                    "quota is used up. Use search_arxiv instead."
                )
            }

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"OpenAlex request failed: {e}. Try search_arxiv."}

    papers = []

    for paper in data.get("results", []):

        best_oa = paper.get("best_oa_location") or {}

        papers.append({
            "title": paper.get("title"),
            "year": paper.get("publication_year"),
            "citations": paper.get("cited_by_count"),
            "doi": paper.get("doi"),
            "abstract": _shorten(
                _rebuild_abstract(paper.get("abstract_inverted_index")),
                MAX_ABSTRACT_CHARS,
            ),
            "pdf_url": best_oa.get("pdf_url"),
        })

    print("[OpenAlex] papers found:", len(papers))

    return papers


# -----------------------------
# Tool 2: arXiv
# -----------------------------

@tool
def search_arxiv(query: str):
    """
    Search arXiv for research papers related to a topic.
    Returns title, authors, published date, abstract and PDF link.
    """

    url = "https://export.arxiv.org/api/query"

    # Quote the phrase so multi-word topics are searched as a phrase
    clean_query = query.replace('"', " ").strip()

    params = {
        "search_query": f'all:"{clean_query}"',
        "start": 0,
        "max_results": 5,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        print("\n[arXiv] status code:", response.status_code)

        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        return {"error": f"arXiv request failed: {e}. Try search_open_alex."}

    feed = feedparser.parse(response.text)

    papers = []

    for entry in feed.entries:

        pdf_url = ""

        for link in entry.get("links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break

        if not pdf_url and entry.get("id"):
            pdf_url = _to_pdf_url(entry.get("id"))

        papers.append({
            "title": _clean_text(entry.get("title", "")),
            "authors": [
                author.get("name", "")
                for author in entry.get("authors", [])
            ][:5],
            "published": entry.get("published", ""),
            "summary": _shorten(
                entry.get("summary", ""),
                MAX_ABSTRACT_CHARS,
            ),
            "pdf_url": pdf_url,
        })

    print("[arXiv] papers found:", len(papers))

    return papers


# -----------------------------
# Tool 3: Read a paper
# -----------------------------

@tool
def read_paper(pdf_url: str):
    """
    Download and read a research paper from a PDF URL.
    Returns the beginning (abstract, introduction) and the end
    (conclusion, limitations, future work) of the paper.
    """

    pdf_url = _to_pdf_url(pdf_url)

    print("\n[read_paper]", pdf_url)

    try:
        response = requests.get(
            pdf_url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        if len(response.content) > MAX_PDF_BYTES:
            return {"error": "PDF is too large to read."}

        reader = PdfReader(BytesIO(response.content))

        pages = []

        for page in reader.pages:
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = ""

            if page_text:
                pages.append(page_text)

        text = "\n".join(pages)

        if not text.strip():
            return {
                "error": (
                    "No text could be extracted (scanned or protected PDF). "
                    "Use the abstract from the search result instead."
                )
            }

        print("[read_paper] pages read:", len(reader.pages))

        return {
            "pdf_url": pdf_url,
            "total_pages": len(reader.pages),
            "text": _select_paper_text(text, MAX_PAPER_CHARS),
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Could not download paper: {e}"}

    except Exception as e:
        return {"error": f"Could not read paper: {e}"}


tools = [
    search_arxiv,
    search_open_alex,
    read_paper,
]