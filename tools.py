import requests
import re

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Research Tools")


# =========================================================
# OPENALEX
# =========================================================

@mcp.tool()
def search_openalex(topic: str) -> list:
    """Search OpenAlex for research papers."""

    url = "https://api.openalex.org/works"

    params = {
        "search": topic,
        "per-page": 5
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    papers = []

    for paper in data.get("results", []):

        papers.append({
            "source": "OpenAlex",
            "title": paper.get("title"),
            "year": paper.get("publication_year"),
            "citations": paper.get("cited_by_count"),
            "doi": paper.get("doi")
        })

    return papers


# =========================================================
# ARXIV
# =========================================================

@mcp.tool()
def search_arxiv(topic: str) -> list:
    """Search arXiv for research papers."""

    url = "https://export.arxiv.org/api/query"

    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": 5,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    xml = response.text

    # -----------------------------------------------------
    # Extract each arXiv entry from XML
    # -----------------------------------------------------

    entries = re.findall(
        r"<entry>(.*?)</entry>",
        xml,
        flags=re.DOTALL
    )

    papers = []

    for entry in entries:

        # -------------------------------------------------
        # TITLE
        # -------------------------------------------------

        title_match = re.search(
            r"<title>(.*?)</title>",
            entry,
            flags=re.DOTALL
        )

        if title_match:

            title = title_match.group(1)

            title = re.sub(
                r"\s+",
                " ",
                title
            ).strip()

        else:

            title = "Unknown arXiv paper"


        # -------------------------------------------------
        # ARXIV ID
        # -------------------------------------------------

        id_match = re.search(
            r"<id>(.*?)</id>",
            entry,
            flags=re.DOTALL
        )

        arxiv_id = None

        if id_match:

            arxiv_id = id_match.group(1).strip()

            # Example:
            # http://arxiv.org/abs/2605.23989v1
            #
            # Get only:
            # 2605.23989v1

            if "/abs/" in arxiv_id:

                arxiv_id = arxiv_id.split(
                    "/abs/"
                )[-1]


        # -------------------------------------------------
        # FALLBACK: FIND ARXIV ID ANYWHERE IN ENTRY
        # -------------------------------------------------

        if not arxiv_id:

            id_match = re.search(
                r"arxiv\.org/(?:abs|pdf)/([^\s<]+)",
                entry,
                flags=re.IGNORECASE
            )

            if id_match:

                arxiv_id = id_match.group(1)


        # -------------------------------------------------
        # CLEAN ID
        # -------------------------------------------------

        if arxiv_id:

            arxiv_id = arxiv_id.strip()

            arxiv_id = arxiv_id.replace(
                ".pdf",
                ""
            )

            arxiv_id = arxiv_id.replace(
                "&amp;",
                ""
            )


        # -------------------------------------------------
        # BUILD PDF URL
        # -------------------------------------------------

        if arxiv_id:

            pdf_url = (
                f"https://arxiv.org/pdf/"
                f"{arxiv_id}.pdf"
            )

        else:

            pdf_url = None


        # -------------------------------------------------
        # AUTHORS
        # -------------------------------------------------

        authors = []

        author_matches = re.findall(
            r"<name>(.*?)</name>",
            entry,
            flags=re.DOTALL
        )

        for author in author_matches:

            authors.append(
                re.sub(
                    r"\s+",
                    " ",
                    author
                ).strip()
            )


        # -------------------------------------------------
        # ONLY ADD VALID PAPER
        # -------------------------------------------------

        if pdf_url:

            papers.append({

                "source": "arXiv",

                "title": title,

                "pdf_url": pdf_url,

                "authors": authors
            })


    return papers


# =========================================================
# MCP SERVER
# =========================================================

if __name__ == "__main__":

    mcp.run(
        transport="stdio"
    )