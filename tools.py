import requests
import re
import xml.etree.ElementTree as ET

from mcp.server.fastmcp import FastMCP


# =========================================================
# MCP SERVER
# =========================================================

mcp = FastMCP(
    "Research Tools"
)


# =========================================================
# ARXIV XML NAMESPACE
# =========================================================

ATOM_NS = "{http://www.w3.org/2005/Atom}"


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

    headers = {
        "User-Agent": "ResearchAgent/1.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    papers = []

    for paper in data.get(
        "results",
        []
    ):

        pdf_url = None

        # -------------------------------------------------
        # BEST OA LOCATION
        # -------------------------------------------------

        best_oa = paper.get(
            "best_oa_location"
        ) or {}

        pdf_url = best_oa.get(
            "pdf_url"
        )

        # -------------------------------------------------
        # PRIMARY LOCATION
        # -------------------------------------------------

        if not pdf_url:

            primary = paper.get(
                "primary_location"
            ) or {}

            pdf_url = primary.get(
                "pdf_url"
            )

        # -------------------------------------------------
        # OTHER LOCATIONS
        # -------------------------------------------------

        if not pdf_url:

            locations = paper.get(
                "locations"
            ) or []

            for location in locations:

                if not isinstance(
                    location,
                    dict
                ):
                    continue

                pdf_url = location.get(
                    "pdf_url"
                )

                if pdf_url:

                    break

        papers.append(
            {
                "source": "OpenAlex",

                "title": paper.get(
                    "title"
                ),

                "year": paper.get(
                    "publication_year"
                ),

                "citations": paper.get(
                    "cited_by_count"
                ),

                "doi": paper.get(
                    "doi"
                ),

                "pdf_url": pdf_url
            }
        )

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

    headers = {
        "User-Agent": "ResearchAgent/1.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    # -----------------------------------------------------
    # PARSE XML
    # -----------------------------------------------------

    try:

        root = ET.fromstring(
            response.content
        )

    except ET.ParseError as e:

        raise ValueError(
            f"Could not parse arXiv response: {e}"
        )

    papers = []

    # -----------------------------------------------------
    # READ ENTRIES
    # -----------------------------------------------------

    for entry in root.findall(
        f"{ATOM_NS}entry"
    ):

        # -------------------------------------------------
        # TITLE
        # -------------------------------------------------

        title_element = entry.find(
            f"{ATOM_NS}title"
        )

        if (
            title_element is not None
            and title_element.text
        ):

            title = re.sub(
                r"\s+",
                " ",
                title_element.text
            ).strip()

        else:

            title = "Unknown arXiv paper"

        # -------------------------------------------------
        # AUTHORS
        # -------------------------------------------------

        authors = []

        for author in entry.findall(
            f"{ATOM_NS}author"
        ):

            name = author.find(
                f"{ATOM_NS}name"
            )

            if (
                name is not None
                and name.text
            ):

                authors.append(
                    name.text.strip()
                )

        # -------------------------------------------------
        # ARXIV ID
        # -------------------------------------------------

        id_element = entry.find(
            f"{ATOM_NS}id"
        )

        arxiv_id = ""

        if (
            id_element is not None
            and id_element.text
        ):

            entry_id = id_element.text.strip()

            if "/abs/" in entry_id:

                arxiv_id = (
                    entry_id
                    .split("/abs/")[-1]
                )

        # -------------------------------------------------
        # PDF LINK
        # -------------------------------------------------

        pdf_url = None

        for link in entry.findall(
            f"{ATOM_NS}link"
        ):

            link_title = link.get(
                "title"
            )

            link_type = link.get(
                "type"
            )

            href = link.get(
                "href"
            )

            if (
                link_title == "pdf"
                or link_type == "application/pdf"
            ):

                pdf_url = href

                break

        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------

        if not pdf_url and arxiv_id:

            pdf_url = (
                "https://arxiv.org/pdf/"
                + arxiv_id
                + ".pdf"
            )

        # -------------------------------------------------
        # ADD PAPER
        # -------------------------------------------------

        papers.append(
            {
                "source": "arXiv",

                "title": title,

                "arxiv_id": arxiv_id,

                "pdf_url": pdf_url,

                "authors": authors
            }
        )

    return papers


# =========================================================
# START MCP SERVER
# =========================================================

if __name__ == "__main__":

    mcp.run(
        transport="stdio"
    )