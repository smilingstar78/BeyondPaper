import os
import json
import re
import requests

from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from paper_reader import store_paper, search_paper
from tools import search_openalex, search_arxiv


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# GROQ MODEL
# =========================================================

_cached_groq_model = None


def get_groq_model():

    global _cached_groq_model

    if _cached_groq_model is not None:
        return _cached_groq_model

    api_key = os.getenv("GROQ_API_KEY")

    print(
        "\nGROQ_API_KEY exists:",
        bool(api_key)
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not available. "
            "Check your local .env file."
        )

    available_models = set()

    try:

        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            timeout=10
        )

        response.raise_for_status()

        models = response.json().get(
            "data",
            []
        )

        available_models = {
            model.get("id")
            for model in models
            if isinstance(model, dict)
        }

        print(
            "Groq models found:",
            len(available_models)
        )

    except Exception as error:

        print(
            "Groq model check failed:",
            error
        )

    preferred_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]

    selected_model = None

    for model in preferred_models:

        if model in available_models:

            selected_model = model
            break

    if selected_model is None:

        selected_model = "llama-3.1-8b-instant"

    print(
        "Using Groq model:",
        selected_model
    )

    _cached_groq_model = ChatGroq(
        model=selected_model,
        api_key=api_key,
        temperature=0
    )

    return _cached_groq_model


# =========================================================
# JSON HELPER
# =========================================================

def parse_json_response(content):

    if not content:
        raise ValueError(
            "Groq returned an empty response."
        )

    content = str(content).strip()

    # Remove markdown code fences
    content = re.sub(
        r"^```json\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"^```\s*",
        "",
        content
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    ).strip()

    # Try direct JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object
    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1 and end > start:

        json_text = content[start:end + 1]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse Groq response as JSON.\n"
        f"Response was:\n{content}"
    )


# =========================================================
# STATE
# =========================================================

class MyState(TypedDict):

    topic: str
    arxiv_papers: list
    open_alex: list
    relevant_papers: list
    paper_analysis: list
    novelty_assessments: list


# =========================================================
# OPENALEX
# =========================================================

async def openalex_node(state: MyState):

    topic = state["topic"]

    print(
        "\n========================================"
    )
    print(
        "SEARCHING OPENALEX"
    )
    print(
        "Topic:",
        topic
    )
    print(
        "========================================"
    )

    try:

        papers = search_openalex(topic)

        if not isinstance(papers, list):
            papers = []

        print(
            "OpenAlex papers found:",
            len(papers)
        )

        return {
            "open_alex": papers
        }

    except Exception as error:

        print(
            "\nOpenAlex ERROR:",
            type(error).__name__,
            error
        )

        return {
            "open_alex": []
        }


# =========================================================
# ARXIV
# =========================================================

async def arxiv_node(state: MyState):

    topic = state["topic"]

    print(
        "\n========================================"
    )
    print(
        "SEARCHING ARXIV"
    )
    print(
        "Topic:",
        topic
    )
    print(
        "========================================"
    )

    try:

        papers = search_arxiv(topic)

        if not isinstance(papers, list):
            papers = []

        print(
            "arXiv papers found:",
            len(papers)
        )

        return {
            "arxiv_papers": papers
        }

    except Exception as error:

        print(
            "\narXiv ERROR:",
            type(error).__name__,
            error
        )

        return {
            "arxiv_papers": []
        }


# =========================================================
# RELEVANT PAPER FINDER
# =========================================================

async def relevant_paper_finder(state: MyState):

    groq = get_groq_model()

    papers = (
        state.get("open_alex", [])
        +
        state.get("arxiv_papers", [])
    )

    print(
        "\n========================================"
    )
    print(
        "PAPER SELECTION"
    )
    print(
        "Total papers:",
        len(papers)
    )
    print(
        "========================================"
    )

    if not papers:

        raise RuntimeError(
            "No papers were returned from OpenAlex or arXiv."
        )

    # Only send useful fields to Groq
    clean_papers = []

    for paper in papers:

        if not isinstance(paper, dict):
            continue

        title = paper.get("title", "")
        pdf_url = paper.get("pdf_url", "")

        if title and pdf_url:

            clean_papers.append({
                "title": title,
                "pdf_url": pdf_url
            })

    if not clean_papers:

        raise RuntimeError(
            "Papers were found, but none contain a usable pdf_url."
        )

    paper_text = json.dumps(
        clean_papers,
        ensure_ascii=False
    )

    prompt = f"""
You are a research paper selection assistant.

Research topic:

{state["topic"]}

Below are research papers retrieved from OpenAlex and arXiv:

{paper_text}

Select the papers that are genuinely relevant to the research topic.

IMPORTANT RULES:

- Select at most 5 papers.
- Select at least 1 paper if relevant papers exist.
- Use the EXACT title from the provided papers.
- Use the EXACT pdf_url from the provided papers.
- Do NOT create URLs.
- Do NOT modify URLs.
- Do NOT invent information.
- Only select papers that contain a usable pdf_url.
- Prefer papers strongly related to the research topic.

Return ONLY valid JSON.

Required format:

{{
    "relevant_papers": [
        {{
            "title": "EXACT paper title",
            "pdf_url": "EXACT pdf_url",
            "reason": "why this paper is relevant"
        }}
    ]
}}
"""

    try:

        response = await groq.ainvoke(prompt)

        content = response.content

        print(
            "\nRelevant paper finder response:"
        )
        print(content)

        result = parse_json_response(content)

    except Exception as error:

        print(
            "\nRELEVANT PAPER FINDER ERROR:"
        )
        print(
            type(error).__name__,
            error
        )

        # IMPORTANT:
        # If Groq selection fails, use the first
        # available papers instead of returning [].
        selected = []

        for paper in clean_papers[:5]:

            selected.append({
                "title": paper["title"],
                "pdf_url": paper["pdf_url"],
                "reason": "Selected from retrieved papers."
            })

        print(
            "\nFallback papers selected:",
            len(selected)
        )

        return {
            "relevant_papers": selected
        }

    selected = []

    for item in result.get(
        "relevant_papers",
        []
    ):

        if not isinstance(item, dict):
            continue

        title = item.get(
            "title",
            ""
        )

        pdf_url = item.get(
            "pdf_url",
            ""
        )

        reason = item.get(
            "reason",
            ""
        )

        if title and pdf_url:

            selected.append({
                "title": title,
                "pdf_url": pdf_url,
                "reason": reason
            })

    # If Groq returned invalid/empty selection,
    # use retrieved papers as fallback.
    if not selected:

        print(
            "\nGroq selected no usable papers."
        )

        for paper in clean_papers[:5]:

            selected.append({
                "title": paper["title"],
                "pdf_url": paper["pdf_url"],
                "reason": "Fallback selection."
            })

    selected = selected[:5]

    print(
        "\nRelevant papers selected:",
        len(selected)
    )

    for paper in selected:

        print(
            "\n-",
            paper["title"]
        )

        print(
            "  PDF:",
            paper["pdf_url"]
        )

    return {
        "relevant_papers": selected
    }


# =========================================================
# PAPER READER
# =========================================================

async def paper_reader_node(state: MyState):

    relevant_papers = state.get(
        "relevant_papers",
        []
    )

    print(
        "\n========================================"
    )
    print(
        "PAPER READER"
    )
    print(
        "Papers:",
        len(relevant_papers)
    )
    print(
        "========================================"
    )

    if not relevant_papers:

        raise RuntimeError(
            "No relevant papers were selected."
        )

    analyses = []

    for paper in relevant_papers:

        title = paper.get(
            "title",
            ""
        )

        pdf_url = paper.get(
            "pdf_url",
            ""
        )

        if not title or not pdf_url:

            print(
                "Skipping paper with missing title/PDF."
            )

            continue

        print(
            "\nReading:",
            title
        )

        print(
            "PDF URL:",
            pdf_url
        )

        try:

            stored = store_paper(
                title=title,
                pdf_url=pdf_url
            )

            print(
                "Paper stored:",
                stored
            )

            if not stored:

                print(
                    "Paper could not be stored."
                )

                continue

            result = search_paper(title)

            if result:

                analyses.append({
                    "title": title,
                    "content": result
                })

                print(
                    "Paper analysis retrieved."
                )

            else:

                print(
                    "No searchable content found."
                )

        except Exception as error:

            print(
                "\nPaper reader error:"
            )

            print(
                type(error).__name__,
                error
            )

            continue

    print(
        "\nTotal papers successfully read:",
        len(analyses)
    )

    if not analyses:

        raise RuntimeError(
            "Relevant papers were selected, "
            "but none could be read/indexed. "
            "Check GEMINI_API_KEY, PDF URLs, "
            "and paper_reader.py."
        )

    return {
        "paper_analysis": analyses
    }


# =========================================================
# RESEARCH AGENT
# =========================================================

async def research_agent_node(state: MyState):

    groq = get_groq_model()

    analyses = state.get(
        "paper_analysis",
        []
    )

    print(
        "\n========================================"
    )
    print(
        "RESEARCH AGENT"
    )
    print(
        "Papers available:",
        len(analyses)
    )
    print(
        "========================================"
    )

    if not analyses:

        raise RuntimeError(
            "Research agent received no paper analysis."
        )

    research_material = json.dumps(
        analyses,
        ensure_ascii=False
    )

    prompt = f"""
You are a research analysis agent.

Research topic:

{state["topic"]}

You have access to information extracted from
existing research papers:

{research_material}

For each relevant paper, identify:

1. Research problem
2. Method used
3. Main limitation
4. Future work
5. Possible research gap

IMPORTANT:

- Use only information supported by the provided paper content.
- Do not invent details.
- If something is not available, say "Not clearly stated".
- Keep the analysis concise.

Return ONLY valid JSON.

Required format:

{{
    "assessments": [
        {{
            "title": "paper title",
            "research_problem": "...",
            "method": "...",
            "limitations": "...",
            "future_work": "...",
            "research_gap": "..."
        }}
    ]
}}
"""

    try:

        response = await groq.ainvoke(prompt)

        content = response.content

        print(
            "\nResearch agent response:"
        )
        print(content)

        result = parse_json_response(content)

    except Exception as error:

        print(
            "\nRESEARCH AGENT ERROR:"
        )

        print(
            type(error).__name__,
            error
        )

        raise RuntimeError(
            f"Research analysis failed: {error}"
        )

    assessments = result.get(
        "assessments",
        []
    )

    if not isinstance(assessments, list):
        assessments = []

    print(
        "\nResearch assessments:",
        len(assessments)
    )

    if not assessments:

        raise RuntimeError(
            "Groq returned no research assessments."
        )

    return {
        "novelty_assessments": assessments
    }


# =========================================================
# GAP ANALYZER
# =========================================================

async def gap_analyzer_node(state: MyState):

    groq = get_groq_model()

    assessments = state.get(
        "novelty_assessments",
        []
    )

    print(
        "\n========================================"
    )
    print(
        "GAP ANALYZER"
    )
    print(
        "Assessments:",
        len(assessments)
    )
    print(
        "========================================"
    )

    if not assessments:

        raise RuntimeError(
            "Gap analyzer received no research assessments."
        )

    material = json.dumps(
        assessments,
        ensure_ascii=False
    )

    prompt = f"""
You are a research gap analysis assistant.

Research topic:

{state["topic"]}

Existing paper analysis:

{material}

Analyze the existing research and identify meaningful
research gaps.

Focus only on gaps supported by the provided papers.

For each gap provide:

- research gap
- why it matters
- possible research direction
- why the direction is different from existing work

IMPORTANT:

- Do not claim that something is novel with certainty.
- Do not invent evidence.
- Base suggestions on the provided research.
- Use careful research language.

Return ONLY valid JSON.

Required format:

{{
    "assessments": [
        {{
            "research_gap": "...",
            "why_it_matters": "...",
            "research_direction": "...",
            "difference_from_existing_work": "..."
        }}
    ]
}}
"""

    try:

        response = await groq.ainvoke(prompt)

        content = response.content

        print(
            "\nGap analyzer response:"
        )
        print(content)

        result = parse_json_response(content)

    except Exception as error:

        print(
            "\nGAP ANALYZER ERROR:"
        )

        print(
            type(error).__name__,
            error
        )

        # Keep the useful research-agent assessments
        # instead of replacing everything with [].
        return {
            "novelty_assessments": assessments
        }

    final_assessments = result.get(
        "assessments",
        []
    )

    if not isinstance(
        final_assessments,
        list
    ):

        final_assessments = []

    if not final_assessments:

        final_assessments = assessments

    print(
        "\nFinal assessments:",
        len(final_assessments)
    )

    return {
        "novelty_assessments": final_assessments
    }


# =========================================================
# BUILD GRAPH
# =========================================================

graph = StateGraph(
    MyState
)


graph.add_node(
    "openalex",
    openalex_node
)

graph.add_node(
    "arxiv",
    arxiv_node
)

graph.add_node(
    "relevant_paper_finder",
    relevant_paper_finder
)

graph.add_node(
    "paper_reader",
    paper_reader_node
)

graph.add_node(
    "research_agent",
    research_agent_node
)

graph.add_node(
    "gap_analyzer",
    gap_analyzer_node
)


# =========================================================
# GRAPH FLOW
# =========================================================

graph.add_edge(
    START,
    "openalex"
)

graph.add_edge(
    "openalex",
    "arxiv"
)

graph.add_edge(
    "arxiv",
    "relevant_paper_finder"
)

graph.add_edge(
    "relevant_paper_finder",
    "paper_reader"
)

graph.add_edge(
    "paper_reader",
    "research_agent"
)

graph.add_edge(
    "research_agent",
    "gap_analyzer"
)

graph.add_edge(
    "gap_analyzer",
    END
)


# =========================================================
# COMPILE
# =========================================================

research_graph = graph.compile()


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    import asyncio

    async def test():

        topic = input(
            "\nEnter research topic: "
        ).strip()

        if not topic:
            return

        initial_state = {
            "topic": topic,
            "arxiv_papers": [],
            "open_alex": [],
            "relevant_papers": [],
            "paper_analysis": [],
            "novelty_assessments": []
        }

        print(
            "\nStarting research graph..."
        )

        try:

            final_state = await research_graph.ainvoke(
                initial_state
            )

            print(
                "\n========================================"
            )
            print(
                "FINAL RESULT"
            )
            print(
                "========================================"
            )

            print(
                json.dumps(
                    final_state.get(
                        "novelty_assessments",
                        []
                    ),
                    indent=2,
                    ensure_ascii=False
                )
            )

        except Exception as error:

            print(
                "\nGRAPH FAILED:"
            )

            print(
                type(error).__name__,
                error
            )

    asyncio.run(test())
