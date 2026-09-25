import os
import json
import re
import requests

from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, START, END

from paper_reader import store_paper, search_paper


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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

    return ChatGroq(
        model=selected_model,
        api_key=api_key,
        temperature=0
    )


TOOLS_PATH = os.path.join(
    BASE_DIR,
    "tools.py"
)


client = MultiServerMCPClient(
    {
        "research_agent": {
            "transport": "stdio",
            "command": "python",
            "args": [TOOLS_PATH]
        }
    }
)


class MyState(TypedDict):

    topic: str

    arxiv_papers: list

    open_alex: list

    relevant_papers: list

    paper_analysis: list

    novelty_assessments: list


async def openalex_node(state: MyState):

    tools = await client.get_tools()

    search_openalex = next(
        tool
        for tool in tools
        if tool.name == "search_openalex"
    )

    papers = await search_openalex.ainvoke(
        {
            "topic": state["topic"]
        }
    )

    if not papers:
        papers = []

    return {
        "open_alex": papers
    }


async def arxiv_node(state: MyState):

    tools = await client.get_tools()

    search_arxiv = next(
        tool
        for tool in tools
        if tool.name == "search_arxiv"
    )

    papers = await search_arxiv.ainvoke(
        {
            "topic": state["topic"]
        }
    )

    if not papers:
        papers = []

    return {
        "arxiv_papers": papers
    }


async def relevant_paper_finder(
    state: MyState
):

    groq = get_groq_model()

    papers = (
        state.get("open_alex", [])
        +
        state.get("arxiv_papers", [])
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

Below are papers retrieved from OpenAlex and arXiv:

{paper_text}

Select the papers that are genuinely relevant to the research topic.

Return ONLY valid JSON in this format:

{{
    "relevant_papers": [
        {{
            "title": "paper title",
            "url": "paper url",
            "reason": "why this paper is relevant"
        }}
    ]
}}

Select at most 5 papers.
"""

    response = await groq.ainvoke(
        prompt
    )

    content = response.content

    try:

        content = re.sub(
            r"```json|```",
            "",
            content
        ).strip()

        result = json.loads(content)

    except Exception:

        result = {
            "relevant_papers": []
        }

    return {
        "relevant_papers": result.get(
            "relevant_papers",
            []
        )
    }


async def paper_reader_node(
    state: MyState
):

    relevant_papers = state.get(
        "relevant_papers",
        []
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

        url = paper.get(
            "url",
            ""
        )

        if not url:
            continue

        try:

            stored = store_paper(
                title=title,
                url=url
            )

            if not stored:
                continue

            result = search_paper(
                title
            )

            if result:

                analyses.append(
                    {
                        "title": title,
                        "content": result
                    }
                )

        except Exception:

            continue

    return {
        "paper_analysis": analyses
    }


async def research_agent_node(
    state: MyState
):

    groq = get_groq_model()

    analyses = state.get(
        "paper_analysis",
        []
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

Do not invent information.

Return ONLY valid JSON:

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

    response = await groq.ainvoke(
        prompt
    )

    content = response.content

    try:

        content = re.sub(
            r"```json|```",
            "",
            content
        ).strip()

        result = json.loads(content)

    except Exception:

        result = {
            "assessments": []
        }

    return {
        "novelty_assessments": result.get(
            "assessments",
            []
        )
    }


async def gap_analyzer_node(
    state: MyState
):

    groq = get_groq_model()

    assessments = state.get(
        "novelty_assessments",
        []
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

Focus on gaps that are supported by the provided papers.

For each gap provide:

- research gap
- why it matters
- possible research direction
- why the direction is different from existing work

Do not claim that something is novel with certainty.
Use careful research language.

Return ONLY valid JSON:

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

    response = await groq.ainvoke(
        prompt
    )

    content = response.content

    try:

        content = re.sub(
            r"```json|```",
            "",
            content
        ).strip()

        result = json.loads(content)

    except Exception:

        result = {
            "assessments": assessments
        }

    return {
        "novelty_assessments": result.get(
            "assessments",
            assessments
        )
    }


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


research_graph = graph.compile()
