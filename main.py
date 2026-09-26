import os
import json
import re
import requests

from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from paper_reader import store_paper, search_paper

from tools import search_openalex, search_arxiv


# =========================================================
# GROQ MODEL
# =========================================================

def get_groq_model():
    api_key = os.environ.get("GROQ_API_KEY")

    print(
        "GROQ_API_KEY exists:",
        bool(api_key)
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not available in the Vercel runtime."
        )

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
        }

    except Exception as error:
        print(
            "Groq model check failed:",
            error
        )

        available_models = set()

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

    return ChatGroq(
        model=selected_model,
        api_key=api_key,
        temperature=0
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

async def openalex_node(
    state: MyState
):
    print(
        "\nSearching OpenAlex for:",
        state["topic"]
    )

    papers = search_openalex(
        state["topic"]
    )

    print(
        "OpenAlex papers found:",
        len(papers)
    )

    return {
        "open_alex": papers
    }


# =========================================================
# ARXIV
# =========================================================

async def arxiv_node(
    state: MyState
):
    print(
        "\nSearching arXiv for:",
        state["topic"]
    )

    papers = search_arxiv(
        state["topic"]
    )

    print(
        "arXiv papers found:",
        len(papers)
    )

    return {
        "arxiv_papers": papers
    }


# =========================================================
# RELEVANT PAPER FINDER
# =========================================================

async def relevant_paper_finder(
    state: MyState
):
    groq = get_groq_model()

    papers = (
        state.get("open_alex", [])
        +
        state.get("arxiv_papers", [])
    )

    print(
        "\nTotal papers before selection:",
        len(papers)
    )

    if not papers:
        return {
            "relevant_papers": []
        }

    paper_text = json.dumps(
        papers,
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
- Use the EXACT title from the provided papers.
- Use the EXACT pdf_url from the provided papers.
- Do NOT create URLs.
- Do NOT modify URLs.
- Do NOT invent information.
- Only select papers that contain a usable pdf_url.
- Prefer papers that are strongly related to the research topic.

Return ONLY valid JSON.

Required format:

{{
    "relevant_papers": [
        {{
            "title": "EXACT paper title",
            "pdf_url": "EXACT pdf_url from provided data",
            "reason": "why this paper is relevant"
        }}
    ]
}}
"""

    try:
        response = await groq.ainvoke(
            prompt
        )

        content = response.content

        print(
            "\nRelevant paper finder response:"
        )
        print(content)

        content = re.sub(
            r"```json|```",
            "",
            content
        ).strip()

        result = json.loads(
            content
        )

    except Exception as error:
        print(
            "\nRelevant paper finder error:",
            error
        )

        result = {
            "relevant_papers": []
        }

    selected = []

    for item in result.get(
        "relevant_papers",
        []
    ):

        if not isinstance(
            item,
            dict
        ):
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

    selected = selected[:5]

    print(
        "\nRelevant papers selected:",
        len(selected)
    )

    for paper in selected:
        print(
            "-",
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

async def paper_reader_node(
    state: MyState
):
    relevant_papers = state.get(
        "relevant_papers",
        []
    )

    print(
        "\nPapers sent to reader:",
        len(relevant_papers)
    )

    if not relevant_papers:
        return {
            "paper_analysis": []
        }

    analyses = []

    for paper in relevant_papers:

        title = paper.get(
            "title",
            ""
        )

        # IMPORTANT:
        # We use pdf_url, not url.
        url = paper.get(
            "pdf_url",
            ""
        )

        print(
            "\nReading:",
            title
        )

        print(
            "PDF URL:",
            url
        )

        if not title or not url:
            print(
                "Skipping paper because title or PDF URL is missing."
            )
            continue

        try:

            stored = store_paper(
                title=title,
                url=url
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

            result = search_paper(
                title
            )

            if result:

                analyses.append({
                    "title": title,
                    "content": result
                })

                print(
                    "Paper analysis data retrieved."
                )

            else:

                print(
                    "No stored content found."
                )

        except Exception as error:

            print(
                "Paper reader error:",
                error
            )

            continue

    print(
        "\nTotal papers successfully read:",
        len(analyses)
    )

    return {
        "paper_analysis": analyses
    }


# =========================================================
# RESEARCH AGENT
# =========================================================

async def research_agent_node(
    state: MyState
):
    groq = get_groq_model()

    analyses = state.get(
        "paper_analysis",
        []
    )

    print(
        "\nPapers available for research analysis:",
        len(analyses)
    )

    if not analyses:
        return {
            "novelty_assessments": []
        }

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

        response = await groq.ainvoke(
            prompt
        )

        content = response.content

        print(
            "\nResearch agent response:"
        )
        print(content)

        content = re.sub(
            r"```json|```",
            "",
            content
        ).strip()

        result = json.loads(
            content
        )

    except Exception as error:

        print(
            "\nResearch agent error:",
            error
        )

        result = {
            "assessments": []
        }

    assessments = result.get(
        "assessments",
        []
    )

    print(
        "\nResearch assessments:",
        len(assessments)
    )

    return {
        "novelty_assessments": assessments
    }


# =========================================================
# GAP ANALYZER
# =========================================================

async def gap_analyzer_node(
    state: MyState
):
    groq = get_groq_model()

    assessments = state.get(
        "novelty_assessments",
        []
    )

    print(
        "\nAssessments sent to gap analyzer:",
        len(assessments)
    )

    if not assessments:
        return {
            "novelty_assessments": []
        }

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
- Base the suggestions on the provided research.
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

        response = await groq.ainvoke(
            prompt
        )

        content = response.content

        print(
            "\nGap analyzer response:"
        )
        print(content)

        content = re.sub(
            r"```json|```",
            "",
            content
        ).strip()

        result = json.loads(
            content
        )

    except Exception as error:

        print(
            "\nGap analyzer error:",
            error
        )

        result = {
            "assessments": assessments
        }

    final_assessments = result.get(
        "assessments",
        assessments
    )

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
