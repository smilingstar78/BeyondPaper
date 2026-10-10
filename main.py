import os
import json
import re
import logging
import requests

from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from paper_reader import store_paper, search_paper
from tools import search_openalex, search_arxiv


# =========================================================
# ENVIRONMENT AND LOGGING
# =========================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# GROQ MODEL
# =========================================================

_cached_groq_model = None


def get_groq_model():
    global _cached_groq_model

    if _cached_groq_model is not None:
        return _cached_groq_model

    api_key = os.getenv("GROQ_API_KEY")

    logger.info(
        "GROQ_API_KEY configured: %s",
        bool(api_key)
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from the environment."
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

        available_models = {
            model.get("id")
            for model in response.json().get("data", [])
            if isinstance(model, dict)
        }

        logger.info(
            "Groq models found: %s",
            len(available_models)
        )

    except Exception:
        logger.exception(
            "Could not retrieve the available Groq models."
        )

    preferred_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]

    selected_model = next(
        (
            model
            for model in preferred_models
            if model in available_models
        ),
        "llama-3.1-8b-instant"
    )

    logger.info("Using Groq model: %s", selected_model)

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

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end > start:
        try:
            return json.loads(
                content[start:end + 1]
            )
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse Groq response as JSON."
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
# OPENALEX NODE
# =========================================================

async def openalex_node(state: MyState):
    topic = state["topic"]

    logger.info(
        "Searching OpenAlex for: %s",
        topic
    )

    try:
        papers = search_openalex(topic)

        if not isinstance(papers, list):
            papers = []

        logger.info(
            "OpenAlex papers found: %s",
            len(papers)
        )

        return {"open_alex": papers}

    except Exception:
        logger.exception("OpenAlex search failed.")
        return {"open_alex": []}


# =========================================================
# ARXIV NODE
# =========================================================

async def arxiv_node(state: MyState):
    topic = state["topic"]

    logger.info(
        "Searching arXiv for: %s",
        topic
    )

    try:
        papers = search_arxiv(topic)

        if not isinstance(papers, list):
            papers = []

        logger.info(
            "arXiv papers found: %s",
            len(papers)
        )

        return {"arxiv_papers": papers}

    except Exception:
        logger.exception("arXiv search failed.")
        return {"arxiv_papers": []}


# =========================================================
# RELEVANT PAPER FINDER
# =========================================================

async def relevant_paper_finder(state: MyState):
    papers = (
        state.get("open_alex", [])
        + state.get("arxiv_papers", [])
    )

    logger.info(
        "Total papers retrieved: %s",
        len(papers)
    )

    if not papers:
        raise RuntimeError(
            "No papers were returned from OpenAlex or arXiv."
        )

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
            "Papers were found, but none contain a usable "
            "title and pdf_url."
        )

    prompt = f"""
You are a research paper selection assistant.

Research topic:
{state["topic"]}

Available papers:
{json.dumps(clean_papers, ensure_ascii=False)}

Select up to 5 papers relevant to the research topic.

Rules:
- Use the exact title and pdf_url provided.
- Never invent or modify URLs.
- Select at least one paper if suitable papers exist.
- Prefer the most relevant papers.

Return only valid JSON in this format:

{{
  "relevant_papers": [
    {{
      "title": "Exact paper title",
      "pdf_url": "Exact PDF URL",
      "reason": "Reason for selection"
    }}
  ]
}}
"""

    try:
        groq = get_groq_model()
        response = await groq.ainvoke(prompt)

        result = parse_json_response(
            response.content
        )

        selected = []

        for item in result.get("relevant_papers", []):
            if not isinstance(item, dict):
                continue

            title = item.get("title", "")
            pdf_url = item.get("pdf_url", "")

            if title and pdf_url:
                selected.append({
                    "title": title,
                    "pdf_url": pdf_url,
                    "reason": item.get("reason", "")
                })

        # Fall back to the retrieved papers if the model
        # fails to return a usable selection.
        if not selected:
            logger.warning(
                "Groq returned no usable paper selection."
            )

            selected = [
                {
                    "title": paper["title"],
                    "pdf_url": paper["pdf_url"],
                    "reason": "Fallback selection."
                }
                for paper in clean_papers[:5]
            ]

        selected = selected[:5]

        logger.info(
            "Relevant papers selected: %s",
            len(selected)
        )

        return {"relevant_papers": selected}

    except Exception:
        logger.exception(
            "Paper selection failed; using fallback papers."
        )

        return {
            "relevant_papers": [
                {
                    "title": paper["title"],
                    "pdf_url": paper["pdf_url"],
                    "reason": "Fallback selection."
                }
                for paper in clean_papers[:5]
            ]
        }


# =========================================================
# PAPER READER NODE
# =========================================================

async def paper_reader_node(state: MyState):
    relevant_papers = state.get(
        "relevant_papers",
        []
    )

    logger.info(
        "Paper reader started. Papers: %s",
        len(relevant_papers)
    )

    if not relevant_papers:
        raise RuntimeError(
            "No relevant papers were selected."
        )

    analyses = []
    failures = []

    for paper in relevant_papers:
        title = paper.get("title", "")
        pdf_url = paper.get("pdf_url", "")

        if not title or not pdf_url:
            failures.append(
                "A selected paper has a missing title or PDF URL."
            )
            continue

        logger.info(
            "Processing paper: %s",
            title
        )

        try:
            stored = store_paper(
                title=title,
                pdf_url=pdf_url
            )

            if not stored:
                failures.append(
                    f"'{title}': PDF reading, embeddings, "
                    "or Chroma storage failed. Check paper_reader "
                    "logs for the detailed exception."
                )
                continue

            result = search_paper(title)

            if not result:
                failures.append(
                    f"'{title}': indexed, but no content "
                    "was retrieved during search."
                )
                continue

            analyses.append({
                "title": title,
                "content": result
            })

            logger.info(
                "Successfully read and indexed: %s",
                title
            )

        except Exception as error:
            logger.exception(
                "Paper reader failed for '%s'.",
                title
            )

            failures.append(
                f"'{title}': {type(error).__name__}: {error}"
            )

    logger.info(
        "Successfully analyzed papers: %s",
        len(analyses)
    )

    if not analyses:
        failure_details = "\n".join(failures)

        raise RuntimeError(
            "Relevant papers were selected, but none could "
            "be read/indexed.\n"
            "Detailed failures:\n"
            f"{failure_details or 'No additional details available.'}"
        )

    return {"paper_analysis": analyses}


# =========================================================
# RESEARCH AGENT
# =========================================================

async def research_agent_node(state: MyState):
    analyses = state.get("paper_analysis", [])

    if not analyses:
        raise RuntimeError(
            "Research agent received no paper analysis."
        )

    prompt = f"""
You are a research analysis assistant.

Research topic:
{state["topic"]}

Extracted paper content:
{json.dumps(analyses, ensure_ascii=False)}

For each paper, identify:
1. Research problem
2. Method used
3. Main limitations
4. Future work
5. Possible research gap

Use only the provided content.
Do not invent details.
If information is missing, say "Not clearly stated".

Return only valid JSON:

{{
  "assessments": [
    {{
      "title": "Paper title",
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
        groq = get_groq_model()
        response = await groq.ainvoke(prompt)

        result = parse_json_response(
            response.content
        )

        assessments = result.get("assessments", [])

        if not isinstance(assessments, list) or not assessments:
            raise ValueError(
                "Groq returned no research assessments."
            )

        return {"novelty_assessments": assessments}

    except Exception as error:
        logger.exception("Research analysis failed.")

        raise RuntimeError(
            f"Research analysis failed: {error}"
        ) from error


# =========================================================
# GAP ANALYZER
# =========================================================

async def gap_analyzer_node(state: MyState):
    assessments = state.get(
        "novelty_assessments",
        []
    )

    if not assessments:
        raise RuntimeError(
            "Gap analyzer received no research assessments."
        )

    prompt = f"""
You are a research gap analysis assistant.

Research topic:
{state["topic"]}

Existing paper assessments:
{json.dumps(assessments, ensure_ascii=False)}

Identify research gaps supported by the provided papers.

For each gap provide:
- research_gap
- why_it_matters
- research_direction
- difference_from_existing_work

Do not claim novelty with certainty.
Do not invent evidence.

Return only valid JSON:

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
        groq = get_groq_model()
        response = await groq.ainvoke(prompt)

        result = parse_json_response(
            response.content
        )

        final_assessments = result.get(
            "assessments",
            []
        )

        if not isinstance(final_assessments, list):
            final_assessments = []

        if not final_assessments:
            final_assessments = assessments

        return {
            "novelty_assessments": final_assessments
        }

    except Exception:
        logger.exception(
            "Gap analysis failed. Keeping previous assessments."
        )

        return {
            "novelty_assessments": assessments
        }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

graph = StateGraph(MyState)

graph.add_node("openalex", openalex_node)
graph.add_node("arxiv", arxiv_node)
graph.add_node(
    "relevant_paper_finder",
    relevant_paper_finder
)
graph.add_node("paper_reader", paper_reader_node)
graph.add_node("research_agent", research_agent_node)
graph.add_node("gap_analyzer", gap_analyzer_node)

graph.add_edge(START, "openalex")
graph.add_edge("openalex", "arxiv")
graph.add_edge("arxiv", "relevant_paper_finder")
graph.add_edge("relevant_paper_finder", "paper_reader")
graph.add_edge("paper_reader", "research_agent")
graph.add_edge("research_agent", "gap_analyzer")
graph.add_edge("gap_analyzer", END)

research_graph = graph.compile()


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        topic = input(
            "Enter research topic: "
        ).strip()

        if not topic:
            print("Research topic is required.")
            return

        initial_state = {
            "topic": topic,
            "arxiv_papers": [],
            "open_alex": [],
            "relevant_papers": [],
            "paper_analysis": [],
            "novelty_assessments": []
        }

        try:
            final_state = await research_graph.ainvoke(
                initial_state
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

        except Exception:
            logger.exception(
                "Research graph failed."
            )

    asyncio.run(test())
