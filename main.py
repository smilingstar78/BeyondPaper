import os
import json
import re
import requests
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END

from paper_reader import store_paper, search_paper


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# DYNAMIC MODEL SELECTOR
# =========================================================

def get_active_groq_model():
    """
    Dynamically find an available text-generation model.

    Whisper/audio models are ignored.
    """

    api_key = os.getenv("GROQ_API_KEY")

    fallback = "llama-3.1-8b-instant"

    if not api_key:
        return fallback

    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}"
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
                and isinstance(m.get("id"), str)
            ]

            print(
                f"\n[INFO] Found "
                f"{len(active_models)} active models "
                f"on your Groq tier."
            )

            # Prefer text-generation models.
            preferred = [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "llama3-8b-8192"
            ]

            for model in preferred:

                if model in active_models:

                    print(
                        f"[INFO] Auto-selected model: "
                        f"{model}\n"
                    )

                    return model

            # Never select audio/vision-only models accidentally.
            blocked_words = [
                "whisper",
                "tts",
                "audio"
            ]

            text_models = [
                model
                for model in active_models
                if not any(
                    word in model.lower()
                    for word in blocked_words
                )
            ]

            if text_models:

                print(
                    f"[INFO] Auto-selected fallback model: "
                    f"{text_models[0]}\n"
                )

                return text_models[0]

    except Exception as e:

        print(
            f"\n[WARNING] Could not fetch models dynamically: "
            f"{e}"
        )

    print(
        f"[INFO] Using fallback model: {fallback}\n"
    )

    return fallback


# =========================================================
# ACTIVE MODEL
# =========================================================

ACTIVE_MODEL = get_active_groq_model()


# =========================================================
# MCP CLIENT
# =========================================================

# DO NOT CHANGE THIS MCP CONFIGURATION.

client = MultiServerMCPClient(
    {
        "research_agent": {
            "transport": "stdio",
            "command": "python",
            "args": [
                r"D:\My Projects\Autonomous Researcher\tools.py"
            ]
        }
    }
)


# =========================================================
# LLMs
# =========================================================

# Small model responses are enough for paper selection.
selection_llm = ChatGroq(
    model=ACTIVE_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=500,
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
)


# Used for extracting structured information from papers.
analysis_llm = ChatGroq(
    model=ACTIVE_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=1000,
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
)


# Used for final research-gap analysis.
gap_llm = ChatGroq(
    model=ACTIVE_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
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

    if isinstance(text, (dict, list)):
        return text

    text = str(text).strip()

    # Direct JSON
    try:
        return json.loads(text)

    except Exception:
        pass

    # Remove markdown JSON fences
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

    # Try extracting JSON object
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
# CONVERT MCP RESULT
# =========================================================

def convert_tool_result(result):

    print(
        "\nDEBUG MCP RESULT TYPE:",
        type(result)
    )

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

                if isinstance(parsed, list):

                    parsed_papers.extend(
                        parsed
                    )

                elif isinstance(parsed, dict):

                    parsed_papers.append(
                        parsed
                    )

            elif hasattr(item, "text"):

                parsed = parse_json(
                    item.text
                )

                if isinstance(parsed, list):

                    parsed_papers.extend(
                        parsed
                    )

                elif isinstance(parsed, dict):

                    parsed_papers.append(
                        parsed
                    )

            elif isinstance(item, str):

                parsed = parse_json(
                    item
                )

                if isinstance(parsed, list):

                    parsed_papers.extend(
                        parsed
                    )

                elif isinstance(parsed, dict):

                    parsed_papers.append(
                        parsed
                    )

        if parsed_papers:

            return parsed_papers

        return result

    if isinstance(result, dict):

        return [result]

    return []


# =========================================================
# NORMALIZE PAPER TITLE
# =========================================================

def normalize_title(title):

    if not isinstance(title, str):

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
# REMOVE DUPLICATE PAPERS
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

            print(
                "\n[INFO] Duplicate paper removed:"
            )

            print(
                f"       {title}"
            )

            continue

        seen_titles.add(
            normalized
        )

        unique_papers.append(
            paper
        )

    return unique_papers


# =========================================================
# OPENALEX NODE
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
            if tool.name == "search_openalex"
        ),
        None
    )

    if search_openalex is None:

        raise ValueError(
            "search_openalex tool was not found."
        )

    result = await search_openalex.ainvoke(
        {
            "topic": state["topic"]
        }
    )

    papers = convert_tool_result(
        result
    )

    if not isinstance(
        papers,
        list
    ):

        papers = []

    print(
        f"OpenAlex found "
        f"{len(papers)} papers."
    )

    return {
        "open_alex": papers
    }


# =========================================================
# ARXIV NODE
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
            if tool.name == "search_arxiv"
        ),
        None
    )

    if search_arxiv is None:

        raise ValueError(
            "search_arxiv tool was not found."
        )

    result = await search_arxiv.ainvoke(
        {
            "topic": state["topic"]
        }
    )

    papers = convert_tool_result(
        result
    )

    if not isinstance(
        papers,
        list
    ):

        papers = []

    print(
        f"arXiv found "
        f"{len(papers)} papers."
    )

    print(
        "\nChecking arXiv PDF URLs...\n"
    )

    for number, paper in enumerate(
        papers,
        start=1
    ):

        print(
            f"{number}. "
            f"{paper.get('title')}"
        )

        print(
            f"   PDF: "
            f"{paper.get('pdf_url')}"
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

    openalex = state.get(
        "open_alex",
        []
    )

    arxiv = state.get(
        "arxiv_papers",
        []
    )

    all_papers = (
        openalex +
        arxiv
    )

    print(
        f"Total papers available: "
        f"{len(all_papers)}"
    )

    # -----------------------------------------------------
    # KEEP ONLY PAPERS WITH PDF URLS
    # -----------------------------------------------------

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

    print(
        f"Papers with PDF URLs: "
        f"{len(pdf_papers)}"
    )

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    pdf_papers = deduplicate_papers(
        pdf_papers
    )

    print(
        f"Unique papers with PDF URLs: "
        f"{len(pdf_papers)}"
    )

    if not pdf_papers:

        print(
            "\nNo papers with PDF URLs found."
        )

        return {
            "relevant_papers": []
        }

    # -----------------------------------------------------
    # CREATE PAPER LIST
    # -----------------------------------------------------

    paper_list = ""

    for index, paper in enumerate(
        pdf_papers
    ):

        paper_list += (
            f"{index}: "
            f"{paper.get('title', 'Unknown')}\n"
        )

    # -----------------------------------------------------
    # PAPER SELECTION PROMPT
    # -----------------------------------------------------

    prompt = f"""
Select the 3 most relevant research papers
for this research topic.

Research topic:
{state["topic"]}

Available papers:
{paper_list}

Choose different papers.

Do not select duplicate or near-duplicate titles.

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

    except Exception as e:

        print(
            f"\nPaper selection failed: {e}"
        )

        print(
            "Using first 3 unique papers instead."
        )

        selected = pdf_papers[:3]

        return {
            "relevant_papers": selected
        }

    # -----------------------------------------------------
    # GET SELECTED INDEXES
    # -----------------------------------------------------

    indexes = result.get(
        "selected_indexes",
        []
    )

    if not isinstance(
        indexes,
        list
    ):

        indexes = []

    selected = []

    for index in indexes:

        try:

            index = int(index)

        except Exception:

            continue

        if (
            0 <= index < len(pdf_papers)
            and pdf_papers[index]
            not in selected
        ):

            selected.append(
                pdf_papers[index]
            )

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    if not selected:

        print(
            "\nLLM did not return valid indexes."
        )

        print(
            "Using first 3 unique papers."
        )

        selected = pdf_papers[:3]

    selected = selected[:3]

    # -----------------------------------------------------
    # PRINT SELECTED PAPERS
    # -----------------------------------------------------

    print(
        "\nSelected papers:"
    )

    for number, paper in enumerate(
        selected,
        start=1
    ):

        print(
            f"\n{number}. "
            f"{paper.get('title')}"
        )

        print(
            f"   PDF: "
            f"{paper.get('pdf_url')}"
        )

    return {
        "relevant_papers": selected
    }


# =========================================================
# PAPER READER NODE
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

            print(
                f"\nSkipping: {title}"
            )

            print(
                "Reason: No PDF URL."
            )

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
                f"\nCould not read: {title}"
            )

            print(
                f"Reason: {e}"
            )

    print(
        f"\nSuccessfully stored "
        f"{len(successful)} papers."
    )

    return {
        "relevant_papers": successful
    }


# =========================================================
# RESEARCH ANALYSIS NODE
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

    # Only retrieve information needed
    # for research-gap analysis.
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

        print(
            f"\nAnalyzing: {title}"
        )

        evidence = {}

        # -------------------------------------------------
        # RETRIEVE RELEVANT EVIDENCE
        # -------------------------------------------------

        for field, question in questions.items():

            try:

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

            except Exception as e:

                evidence[field] = (
                    f"Retrieval error: {e}"
                )

        # -------------------------------------------------
        # BUILD COMPACT EVIDENCE
        # -------------------------------------------------

        evidence_text = ""

        for field, text in evidence.items():

            evidence_text += (
                f"\n\n{field.upper()}:\n"
                f"{str(text)[:1000]}"
            )

        # -------------------------------------------------
        # PAPER ANALYSIS PROMPT
        # -------------------------------------------------

        prompt = f"""
Analyze this research paper using ONLY
the retrieved evidence.

Do not use outside knowledge.
Do not invent information.

Paper Title:
{title}

Retrieved evidence:
{evidence_text}

Keep every answer concise.

Return ONLY one valid JSON object.

Format:

{{
    "research_problem": "brief answer",
    "method": "brief answer",
    "limitations": "brief answer",
    "future_work": "brief answer"
}}

If information is missing, write:

"Not found in retrieved evidence."
"""

        try:

            response = await analysis_llm.ainvoke(
                prompt
            )

            analysis = parse_json(
                response.content
            )

            if (
                not isinstance(
                    analysis,
                    dict
                )
                or not analysis
            ):

                analysis = {
                    "error":
                        "Failed to parse JSON",
                    "raw_response":
                        response.content
                }

            analysis["paper"] = title

            all_analysis.append(
                analysis
            )

        except Exception as e:

            print(
                f"Analysis failed: {e}"
            )

            all_analysis.append(
                {
                    "paper": title,
                    "error": str(e)
                }
            )

    print(
        f"\nSuccessfully analyzed "
        f"{len(all_analysis)} papers."
    )

    return {
        "paper_analysis":
            all_analysis
    }


# =========================================================
# RESEARCH GAP & NOVELTY CHECK NODE
# =========================================================

async def gap_analyzer_node(state):

    print(
        "\nEvaluating Literature & "
        "Generating Novelty Assessments..."
    )

    topic = state.get(
        "topic",
        ""
    )

    analyses = state.get(
        "paper_analysis",
        []
    )

    # -----------------------------------------------------
    # SEARCH COVERAGE
    # -----------------------------------------------------

    openalex_count = len(
        state.get(
            "open_alex",
            []
        )
    )

    arxiv_count = len(
        state.get(
            "arxiv_papers",
            []
        )
    )

    total_retrieved = (
        openalex_count +
        arxiv_count
    )

    # -----------------------------------------------------
    # COUNT PAPERS WITH PDF
    # -----------------------------------------------------

    all_retrieved_papers = (
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

    for paper in all_retrieved_papers:

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

    unique_pdf_papers = deduplicate_papers(
        pdf_papers
    )

    pdf_count = len(
        unique_pdf_papers
    )

    # -----------------------------------------------------
    # COUNT SUCCESSFULLY ANALYZED PAPERS
    # -----------------------------------------------------

    analyzed_count = len(
        [
            item
            for item in analyses
            if (
                isinstance(
                    item,
                    dict
                )
                and "error" not in item
            )
        ]
    )

    if not analyses:

        return {
            "novelty_assessments": []
        }

    # -----------------------------------------------------
    # PREPARE COMPACT PAPER SUMMARIES
    # -----------------------------------------------------

    summaries_text = ""

    for idx, item in enumerate(
        analyses,
        start=1
    ):

        if "error" in item:

            continue

        paper_title = item.get(
            "paper",
            "Unknown"
        )

        summaries_text += f"""

PAPER {idx}: {paper_title}

Research Problem:
{str(
    item.get(
        "research_problem",
        "N/A"
    )
)[:600]}

Method:
{str(
    item.get(
        "method",
        "N/A"
    )
)[:600]}

Limitations:
{str(
    item.get(
        "limitations",
        "N/A"
    )
)[:600]}

Future Work:
{str(
    item.get(
        "future_work",
        "N/A"
    )
)[:600]}
"""

    # -----------------------------------------------------
    # GAP ANALYSIS PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are an expert AI research advisor.

Research topic:
{topic}

Search coverage:
OpenAlex: {openalex_count}
arXiv: {arxiv_count}
Total retrieved: {total_retrieved}
Unique PDFs: {pdf_count}
Papers analyzed: {analyzed_count}

Retrieved literature evidence:
{summaries_text}

Generate 2 to 3 candidate research topics
based ONLY on gaps supported by the retrieved
literature.

For each candidate:

1. Give a specific research topic.
2. Explain similarity with the retrieved papers.
3. Explain the remaining research gap.
4. List the supporting evidence papers.
5. Assign a status.

Do NOT claim definitive novelty.

Do NOT claim that no previous research exists.

Do NOT invent papers, methods, results,
or research gaps.

Use cautious wording such as:

"The retrieved literature does not clearly demonstrate..."

"The analyzed papers do not directly address..."

"The retrieved evidence provides limited coverage of..."

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

    # -----------------------------------------------------
    # RUN FINAL LLM
    # -----------------------------------------------------

    try:

        response = await gap_llm.ainvoke(
            prompt
        )

        res_json = parse_json(
            response.content
        )

        assessments = res_json.get(
            "assessments",
            []
        )

        if not isinstance(
            assessments,
            list
        ):

            assessments = []

        # -------------------------------------------------
        # PROCESS EACH ASSESSMENT
        # -------------------------------------------------

        for item in assessments:

            if not isinstance(
                item,
                dict
            ):

                continue

            # ---------------------------------------------
            # EVIDENCE PAPERS
            # ---------------------------------------------

            item_evidence = item.get(
                "evidence_papers",
                []
            )

            if not isinstance(
                item_evidence,
                list
            ):

                item_evidence = []

            # ---------------------------------------------
            # REMOVE DUPLICATE EVIDENCE NAMES
            # ---------------------------------------------

            clean_evidence = []

            seen_evidence = set()

            for paper in item_evidence:

                if not isinstance(
                    paper,
                    str
                ):

                    continue

                normalized = normalize_title(
                    paper
                )

                if (
                    normalized
                    and normalized
                    not in seen_evidence
                ):

                    seen_evidence.add(
                        normalized
                    )

                    clean_evidence.append(
                        paper
                    )

            item[
                "evidence_papers"
            ] = clean_evidence

            # ---------------------------------------------
            # CALCULATE CONFIDENCE
            # ---------------------------------------------

            evidence_count = len(
                clean_evidence
            )

            if evidence_count <= 1:

                confidence = "Low"

            elif evidence_count <= 3:

                confidence = "Medium"

            else:

                confidence = "High"

            item[
                "evidence_confidence"
            ] = confidence

            # ---------------------------------------------
            # ADD SEARCH COVERAGE
            # ---------------------------------------------

            item[
                "search_coverage"
            ] = {

                "openalex_papers":
                    openalex_count,

                "arxiv_papers":
                    arxiv_count,

                "total_retrieved":
                    total_retrieved,

                "papers_with_pdf":
                    pdf_count,

                "papers_analyzed":
                    analyzed_count
            }

            # ---------------------------------------------
            # ADD COVERAGE LIMITATION
            # ---------------------------------------------

            item[
                "coverage_limitation"
            ] = (
                "This assessment is based on "
                f"{analyzed_count} analyzed papers "
                f"from {total_retrieved} retrieved papers "
                "and may not represent all existing "
                "research on the topic."
            )

        return {
            "novelty_assessments":
                assessments
        }

    except Exception as e:

        print(
            f"\n[ERROR] Novelty Assessment failed: "
            f"{e}"
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


app = graph.compile()


# =========================================================
# MAIN
# =========================================================

async def main():

    topic = input(
        "\nEnter research topic: "
    ).strip()

    if not topic:

        print(
            "Please enter a research topic."
        )

        return

    initial_state = {

        "topic": topic,

        "arxiv_papers": [],

        "open_alex": [],

        "relevant_papers": [],

        "paper_analysis": [],

        "novelty_assessments": []
    }

    final_state = await app.ainvoke(
        initial_state
    )

    assessments = final_state.get(
        "novelty_assessments",
        []
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EXISTING-WORK & RESEARCH-GAP VERIFICATION"
    )

    print(
        "=" * 70
    )

    if not assessments:

        print(
            "\nNo research-gap assessments generated."
        )

        return

    for item in assessments:

        print(
            f"\nTopic "
            f"{item.get('topic_number', 1)}:"
        )

        print(
            f"{item.get('topic_title', '')}\n"
        )

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        print(
            "Status:"
        )

        print(
            f"{item.get(
                'status',
                'Potentially underexplored'
            )}\n"
        )

        # -------------------------------------------------
        # SIMILARITY ANALYSIS
        # -------------------------------------------------

        print(
            "Similarity Analysis:"
        )

        print(
            f"{item.get(
                'similarity_analysis',
                ''
            )}\n"
        )

        # -------------------------------------------------
        # REMAINING GAP
        # -------------------------------------------------

        print(
            "Remaining Gap:"
        )

        print(
            f"{item.get(
                'remaining_gap',
                ''
            )}\n"
        )

        # -------------------------------------------------
        # SEARCH COVERAGE
        # -------------------------------------------------

        coverage = item.get(
            "search_coverage",
            {}
        )

        print(
            "Search Coverage:"
        )

        print(
            f"OpenAlex papers retrieved: "
            f"{coverage.get(
                'openalex_papers',
                0
            )}"
        )

        print(
            f"arXiv papers retrieved: "
            f"{coverage.get(
                'arxiv_papers',
                0
            )}"
        )

        print(
            f"Total papers retrieved: "
            f"{coverage.get(
                'total_retrieved',
                0
            )}"
        )

        print(
            f"Unique papers with PDF URLs: "
            f"{coverage.get(
                'papers_with_pdf',
                0
            )}"
        )

        print(
            f"Papers analyzed in depth: "
            f"{coverage.get(
                'papers_analyzed',
                0
            )}\n"
        )

        # -------------------------------------------------
        # EVIDENCE CONFIDENCE
        # -------------------------------------------------

        print(
            "Evidence Confidence:"
        )

        print(
            f"{item.get(
                'evidence_confidence',
                'Low'
            )}\n"
        )

        # -------------------------------------------------
        # EVIDENCE PAPERS
        # -------------------------------------------------

        print(
            "Evidence Papers:"
        )

        evidence_papers = item.get(
            "evidence_papers",
            []
        )

        if evidence_papers:

            for paper in evidence_papers:

                print(
                    f"- {paper}"
                )

        else:

            print(
                "- No specific evidence papers returned."
            )

        # -------------------------------------------------
        # COVERAGE LIMITATION
        # -------------------------------------------------

        print(
            "\nCoverage Limitation:"
        )

        print(
            item.get(
                "coverage_limitation",
                "The assessment is based on the retrieved literature and may not represent all existing research on the topic."
            )
        )

        # -------------------------------------------------
        # NOVELTY NOTE
        # -------------------------------------------------

        print(
            "\nNote:"
        )

        print(
            "This assessment is based on the retrieved "
            "literature and does not establish definitive "
            "research novelty."
        )

        print(
            "-" * 70
        )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    import asyncio

    asyncio.run(
        main()
    )