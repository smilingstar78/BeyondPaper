import os
import json
import re
import requests

from typing import TypedDict

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END

from paper_reader import (
    store_paper,
    search_paper
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# DYNAMIC GROQ MODEL
# =========================================================

def get_active_groq_model():

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    fallback = "llama-3.1-8b-instant"

    if not api_key:
        return fallback

    try:

        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization":
                    f"Bearer {api_key}"
            },
            timeout=5
        )

        if response.status_code == 200:

            models = response.json().get(
                "data",
                []
            )

            active_models = [
                m["id"]
                for m in models
                if isinstance(m, dict)
                and isinstance(
                    m.get("id"),
                    str
                )
            ]

            preferred = [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b"
            ]

            for model in preferred:

                if model in active_models:

                    print(
                        f"[INFO] "
                        f"Using Groq model: "
                        f"{model}"
                    )

                    return model

    except Exception as e:

        print(
            f"[WARNING] "
            f"Could not fetch Groq models: "
            f"{e}"
        )

    return fallback


# =========================================================
# ACTIVE MODEL
# =========================================================

ACTIVE_MODEL = get_active_groq_model()


# =========================================================
# MCP CLIENT
# =========================================================

TOOLS_PATH = os.path.join(
    BASE_DIR,
    "tools.py"
)

client = MultiServerMCPClient(
    {
        "research_agent": {
            "transport": "stdio",
            "command": "python",
            "args": [
                TOOLS_PATH
            ]
        }
    }
)


# =========================================================
# LLMs
# =========================================================

selection_llm = ChatGroq(
    model=ACTIVE_MODEL,
    api_key=os.getenv(
        "GROQ_API_KEY"
    ),
    max_tokens=500,
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
)


analysis_llm = ChatGroq(
    model=ACTIVE_MODEL,
    api_key=os.getenv(
        "GROQ_API_KEY"
    ),
    max_tokens=1000,
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
)


gap_llm = ChatGroq(
    model=ACTIVE_MODEL,
    api_key=os.getenv(
        "GROQ_API_KEY"
    ),
    max_tokens=1200,
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
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
# JSON PARSER
# =========================================================

def parse_json(text):

    if not text:
        return {}

    if isinstance(
        text,
        (dict, list)
    ):
        return text

    text = str(text).strip()

    try:
        return json.loads(text)

    except Exception:
        pass

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()

    try:
        return json.loads(text)

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        try:

            return json.loads(
                text[start:end + 1]
            )

        except Exception:
            pass

    return {}


# =========================================================
# MCP RESULT CONVERTER
# =========================================================

def convert_tool_result(result):

    parsed_papers = []

    if isinstance(result, str):

        parsed = parse_json(result)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            return [parsed]

        return []

    if hasattr(result, "content"):

        return convert_tool_result(
            result.content
        )

    if isinstance(result, list):

        for item in result:

            if (
                isinstance(item, dict)
                and (
                    "title" in item
                    or "pdf_url" in item
                )
            ):

                parsed_papers.append(item)

            elif (
                isinstance(item, dict)
                and "text" in item
            ):

                parsed = parse_json(
                    item["text"]
                )

                if isinstance(
                    parsed,
                    list
                ):

                    parsed_papers.extend(
                        parsed
                    )

                elif isinstance(
                    parsed,
                    dict
                ):

                    parsed_papers.append(
                        parsed
                    )

            elif hasattr(item, "text"):

                parsed = parse_json(
                    item.text
                )

                if isinstance(
                    parsed,
                    list
                ):

                    parsed_papers.extend(
                        parsed
                    )

                elif isinstance(
                    parsed,
                    dict
                ):

                    parsed_papers.append(
                        parsed
                    )

            elif isinstance(
                item,
                str
            ):

                parsed = parse_json(item)

                if isinstance(
                    parsed,
                    list
                ):

                    parsed_papers.extend(
                        parsed
                    )

                elif isinstance(
                    parsed,
                    dict
                ):

                    parsed_papers.append(
                        parsed
                    )

        return parsed_papers

    if isinstance(result, dict):
        return [result]

    return []


# =========================================================
# TITLE NORMALIZATION
# =========================================================

def normalize_title(title):

    if not isinstance(
        title,
        str
    ):
        return ""

    title = title.lower().strip()

    title = re.sub(
        r"[^a-z0-9\s]",
        "",
        title
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    return title


# =========================================================
# DEDUPLICATION
# =========================================================

def deduplicate_papers(papers):

    unique_papers = []

    seen_titles = set()

    for paper in papers:

        if not isinstance(
            paper,
            dict
        ):
            continue

        title = paper.get(
            "title",
            ""
        )

        normalized = normalize_title(
            title
        )

        if not normalized:
            continue

        if normalized in seen_titles:
            continue

        seen_titles.add(
            normalized
        )

        unique_papers.append(
            paper
        )

    return unique_papers


# =========================================================
# OPENALEX
# =========================================================

async def openalex_node(state):

    print(
        "\nSearching OpenAlex..."
    )

    tools = await client.get_tools()

    search_openalex = next(
        (
            tool
            for tool in tools
            if tool.name ==
            "search_openalex"
        ),
        None
    )

    if search_openalex is None:

        raise ValueError(
            "search_openalex tool "
            "was not found."
        )

    result = await search_openalex.ainvoke(
        {
            "topic":
                state["topic"]
        }
    )

    papers = convert_tool_result(
        result
    )

    return {
        "open_alex": papers
    }


# =========================================================
# ARXIV
# =========================================================

async def arxiv_node(state):

    print(
        "\nSearching arXiv..."
    )

    tools = await client.get_tools()

    search_arxiv = next(
        (
            tool
            for tool in tools
            if tool.name ==
            "search_arxiv"
        ),
        None
    )

    if search_arxiv is None:

        raise ValueError(
            "search_arxiv tool "
            "was not found."
        )

    result = await search_arxiv.ainvoke(
        {
            "topic":
                state["topic"]
        }
    )

    papers = convert_tool_result(
        result
    )

    return {
        "arxiv_papers": papers
    }


# =========================================================
# FIND RELEVANT PAPERS
# =========================================================

async def relevant_paper_finder(state):

    print(
        "\nFinding relevant papers..."
    )

    all_papers = (
        state.get(
            "open_alex",
            []
        )
        +
        state.get(
            "arxiv_papers",
            []
        )
    )

    pdf_papers = []

    for paper in all_papers:

        if not isinstance(
            paper,
            dict
        ):
            continue

        pdf_url = paper.get(
            "pdf_url"
        )

        if (
            isinstance(
                pdf_url,
                str
            )
            and pdf_url.strip()
        ):

            pdf_papers.append(
                paper
            )

    pdf_papers = deduplicate_papers(
        pdf_papers
    )

    if not pdf_papers:

        return {
            "relevant_papers": []
        }

    paper_list = ""

    for index, paper in enumerate(
        pdf_papers
    ):

        paper_list += (
            f"{index}: "
            f"{paper.get('title', 'Unknown')}\n"
        )

    prompt = f"""
Select the 3 most relevant research papers
for this research topic.

Research topic:

{state["topic"]}

Available papers:

{paper_list}

Choose different papers.

Return ONLY valid JSON.

Format:

{{
    "selected_indexes": [0, 1, 2]
}}
"""

    try:

        response = await selection_llm.ainvoke(
            prompt
        )

        result = parse_json(
            response.content
        )

    except Exception:

        return {
            "relevant_papers":
                pdf_papers[:3]
        }

    indexes = result.get(
        "selected_indexes",
        []
    )

    selected = []

    for index in indexes:

        try:
            index = int(index)
        except Exception:
            continue

        if (
            0 <= index <
            len(pdf_papers)
        ):

            if pdf_papers[index] not in selected:

                selected.append(
                    pdf_papers[index]
                )

    if not selected:

        selected = pdf_papers[:3]

    return {
        "relevant_papers":
            selected[:3]
    }


# =========================================================
# PAPER READER
# =========================================================

async def paper_reader_node(state):

    print(
        "\nReading selected papers..."
    )

    papers = state.get(
        "relevant_papers",
        []
    )

    successful = []

    for paper in papers:

        title = paper.get(
            "title",
            "Unknown"
        )

        pdf_url = paper.get(
            "pdf_url"
        )

        if not pdf_url:
            continue

        try:

            store_paper(
                title,
                pdf_url
            )

            successful.append(
                paper
            )

        except Exception as e:

            print(
                f"Could not read "
                f"{title}: {e}"
            )

    return {
        "relevant_papers":
            successful
    }


# =========================================================
# PAPER ANALYSIS
# =========================================================

async def research_agent_node(state):

    print(
        "\nAnalyzing papers..."
    )

    papers = state.get(
        "relevant_papers",
        []
    )

    all_analysis = []

    questions = {

        "research_problem":
            "What is the research problem, motivation, or research gap?",

        "method":
            "What methodology, model, algorithm, or approach is proposed?",

        "limitations":
            "What limitations, weaknesses, or challenges are mentioned?",

        "future_work":
            "What future work or improvements are suggested?"
    }

    for paper in papers:

        title = paper.get(
            "title",
            "Unknown"
        )

        evidence = {}

        for field, question in questions.items():

            chunks = search_paper(
                question,
                title,
                k=2
            )

            if chunks:

                evidence[field] = (
                    "\n".join(
                        chunks[:2]
                    )[:1000]
                )

            else:

                evidence[field] = (
                    "Not found."
                )

        evidence_text = ""

        for field, text in evidence.items():

            evidence_text += (
                f"\n\n{field.upper()}:\n"
                f"{str(text)[:1000]}"
            )

        prompt = f"""
Analyze this research paper using ONLY
the retrieved evidence.

Do not use outside knowledge.

Paper Title:

{title}

Retrieved evidence:

{evidence_text}

Return ONLY valid JSON.

Format:

{{
    "research_problem": "brief answer",
    "method": "brief answer",
    "limitations": "brief answer",
    "future_work": "brief answer"
}}
"""

        try:

            response = await analysis_llm.ainvoke(
                prompt
            )

            analysis = parse_json(
                response.content
            )

            if not isinstance(
                analysis,
                dict
            ):

                analysis = {}

            analysis["paper"] = title

            all_analysis.append(
                analysis
            )

        except Exception as e:

            all_analysis.append(
                {
                    "paper": title,
                    "error": str(e)
                }
            )

    return {
        "paper_analysis":
            all_analysis
    }


# =========================================================
# GAP ANALYZER
# =========================================================

async def gap_analyzer_node(state):

    print(
        "\nGenerating research-gap analysis..."
    )

    topic = state.get(
        "topic",
        ""
    )

    analyses = state.get(
        "paper_analysis",
        []
    )

    if not analyses:

        return {
            "novelty_assessments": []
        }

    summaries_text = ""

    for idx, item in enumerate(
        analyses,
        start=1
    ):

        if "error" in item:
            continue

        summaries_text += f"""

PAPER {idx}: {item.get("paper", "Unknown")}

Research Problem:
{str(item.get("research_problem", "N/A"))[:600]}

Method:
{str(item.get("method", "N/A"))[:600]}

Limitations:
{str(item.get("limitations", "N/A"))[:600]}

Future Work:
{str(item.get("future_work", "N/A"))[:600]}
"""

    prompt = f"""
You are an expert AI research advisor.

Research topic:

{topic}

Retrieved literature evidence:

{summaries_text}

Generate 2 to 3 candidate research topics
based ONLY on gaps supported by the retrieved
literature.

Do not claim definitive novelty.

Do not claim that no previous research exists.

Do not invent papers, methods, results,
or research gaps.

Use cautious wording.

Status MUST be exactly one of:

"Potentially underexplored"

"Partially addressed"

"High overlap in literature"

Return ONLY valid JSON.

Format:

{{
    "assessments": [
        {{
            "topic_number": 1,
            "topic_title": "Specific research topic",
            "status": "Potentially underexplored",
            "similarity_analysis": "Evidence-based comparison.",
            "remaining_gap": "Evidence-based remaining gap.",
            "evidence_papers": ["Paper title"]
        }}
    ]
}}
"""

    try:

        response = await gap_llm.ainvoke(
            prompt
        )

        result = parse_json(
            response.content
        )

        assessments = result.get(
            "assessments",
            []
        )

        return {
            "novelty_assessments":
                assessments
        }

    except Exception as e:

        print(
            f"Gap analysis failed: {e}"
        )

        return {
            "novelty_assessments": []
        }


# =========================================================
# GRAPH
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
# COMPILED RESEARCH GRAPH
# =========================================================

research_graph = graph.compile()
