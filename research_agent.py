from analyze_papers import MyState
import os
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

class PaperAnalysis(BaseModel):
    research_problem: str
    method: str
    dataset: str
    results: str
    metrics: str
    limitations: str
    future_work: str

class OutputSchema(BaseModel):

    result: Literal["arxiv", "alex", "read", "finish"] = Field(
        description="The next action for the research agent."
    )


structured_model = model.with_structured_output(
    OutputSchema,
    method="json_mode"
)

structured_model2 = model.with_structured_output(
    PaperAnalysis,
    method="json_mode"
)


def research_agent(state: MyState):

    prompt = f"""
You are a research agent.

Research topic:
{state["topic"]}

OpenAlex papers found:
{len(state["alex_papers"])}

arXiv papers found:
{len(state["arxiv_papers"])}

Papers already read:
{len(state["paper_texts"])}

Decide what should be done next.

Available actions:

- "alex" → search OpenAlex for papers
- "arxiv" → search arXiv for papers
- "read" → read the papers that have been found
- "finish" → finish the research phase

Rules:

- If both OpenAlex and arXiv have 0 papers, choose "alex".
- If OpenAlex has papers but arXiv has 0 papers, choose "arxiv".
- If papers have been found but paper_texts is empty, choose "read".
- If papers have already been read, choose "finish".

Return the result as JSON.

The JSON must have this format:

{{"result": "alex"}}

The result must be exactly one of:

"alex"
"arxiv"
"read"
"finish"
"""

    result = structured_model.invoke(prompt)

    print("\nAgent decision:", result.result)

    return {
        "next_action": result.result
    }


def decision(state: MyState):

    return state["next_action"]

def research_gaps(state: MyState):

    analysis = state["paper_analysis"]

    prompt = f"""
You are a research gap analyst.

Research topic:
{state["topic"]}

Here are the analyses of the research papers:

{analysis}

Compare these papers and identify research gaps.

Look especially at:

- repeated limitations
- missing datasets
- missing evaluation
- weaknesses in existing methods
- unexplored research areas
- future work mentioned by multiple papers

Only identify gaps that are supported by the provided paper analyses.
Do not invent information.
"""

    result = model.invoke(prompt)

    print("\nResearch gaps:", result.content)

    return {
        "research_gaps": result.content
    }


def analyze_papers(state: MyState):

    paper_texts = state["paper_texts"]

    paper_analysis = []

    for paper in paper_texts:

        prompt = f"""
Analyze this research paper.

Title:
{paper["title"]}

Paper text:
{paper["text"][:12000]}

Extract the following information.

IMPORTANT:
Return ONLY valid JSON.

The JSON keys MUST be exactly these:

"research_problem"
"method"
"dataset"
"results"
"metrics"
"limitations"
"future_work"

Do NOT use:
"ResearchProblem"
"Research Problem"
"Method"
"Dataset"
"Results"
"Metrics"
"Limitations"
"FutureWork"

Use the exact lowercase snake_case keys shown above.

Return this exact structure:

{{
    "research_problem": "...",
    "method": "...",
    "dataset": "...",
    "results": "...",
    "metrics": "...",
    "limitations": "...",
    "future_work": "..."
}}
"""

        result = structured_model2.invoke(prompt)

        paper_analysis.append({
            "title": paper["title"],
            "research_problem": result.research_problem,
            "method": result.method,
            "dataset": result.dataset,
            "results": result.results,
            "metrics": result.metrics,
            "limitations": result.limitations,
            "future_work": result.future_work
        })

    return {
        "paper_analysis": paper_analysis
    }