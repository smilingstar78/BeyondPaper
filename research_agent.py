from analyze_papers import MyState

import os
import json

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


# =========================
# MODELS
# =========================

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

model2 = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)


# =========================
# SCHEMAS
# =========================

class PaperAnalysis(BaseModel):
    research_problem: str
    method: str
    dataset: str
    results: str
    metrics: str
    limitations: str
    future_work: str
    evidence: str


class OutputSchema(BaseModel):
    result: Literal[
        "arxiv",
        "alex",
        "read",
        "finish"
    ] = Field(
        description="The next action for the research agent."
    )


# =========================
# STRUCTURED MODEL
# =========================

structured_model = model.with_structured_output(
    OutputSchema,
    method="json_mode"
)


# =========================
# RESEARCH AGENT
# =========================

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

- "alex" → search OpenAlex
- "arxiv" → search arXiv
- "read" → read the papers
- "finish" → finish the research phase

Rules:

- If both OpenAlex and arXiv have 0 papers, choose "alex".
- If OpenAlex has papers but arXiv has 0 papers, choose "arxiv".
- If papers have been found but paper_texts is empty, choose "read".
- If papers have already been read, choose "finish".

Return ONLY valid JSON.

Use exactly this key:

{{
    "result": "alex"
}}

The value must be exactly one of:

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


# =========================
# ANALYZE PAPERS
# =========================

def analyze_papers(state: MyState):

    paper_texts = state["paper_texts"]

    paper_analysis = []

    for paper in paper_texts:

        print("\nAnalyzing:", paper["title"])

        prompt = f"""
You are a research paper analysis system.

Analyze the following research paper.

TITLE:
{paper["title"]}

PAPER TEXT:
{paper["text"][:12000]}

Extract these eight fields:

1. research_problem
2. method
3. dataset
4. results
5. metrics
6. limitations
7. future_work
8. evidence

IMPORTANT RULES:

- Use ONLY information from the provided paper text.
- Do NOT invent information.
- If information is not mentioned, write "Not mentioned".
- Evidence must be based on the provided paper text.
- Keep each field reasonably concise.
- Do NOT use different field names.
- Do NOT capitalize the field names.
- Do NOT add extra fields.

Return ONLY valid JSON.

Use EXACTLY this structure:

{{
    "research_problem": "...",
    "method": "...",
    "dataset": "...",
    "results": "...",
    "metrics": "...",
    "limitations": "...",
    "future_work": "...",
    "evidence": "..."
}}
"""

        result = model2.invoke(prompt)

        # Get the model's text response
        content = result.content

        # Convert JSON text into Python dictionary
        try:

            data = json.loads(content)

        except json.JSONDecodeError:

            print("\nJSON parsing failed for:", paper["title"])
            print("Model response:")
            print(content)

            continue


        # Handle possible capitalization / spacing differences
        normalized_data = {}

        for key, value in data.items():

            clean_key = key.strip().lower().replace(" ", "_")

            normalized_data[clean_key] = value


        # Make sure all expected fields exist
        paper_result = {
            "title": paper["title"],

            "research_problem": normalized_data.get(
                "research_problem",
                "Not mentioned"
            ),

            "method": normalized_data.get(
                "method",
                "Not mentioned"
            ),

            "dataset": normalized_data.get(
                "dataset",
                "Not mentioned"
            ),

            "results": normalized_data.get(
                "results",
                "Not mentioned"
            ),

            "metrics": normalized_data.get(
                "metrics",
                "Not mentioned"
            ),

            "limitations": normalized_data.get(
                "limitations",
                "Not mentioned"
            ),

            "future_work": normalized_data.get(
                "future_work",
                "Not mentioned"
            ),

            "evidence": normalized_data.get(
                "evidence",
                "Not mentioned"
            )
        }

        paper_analysis.append(paper_result)

        print("Successfully analyzed:", paper["title"])


    print("\nTotal papers analyzed:", len(paper_analysis))

    return {
        "paper_analysis": paper_analysis
    }


# =========================
# RESEARCH GAPS
# =========================

def research_gaps(state: MyState):

    gap_data = []

    for paper in state["paper_analysis"]:

        gap_data.append({
            "title": paper["title"],
            "limitations": paper["limitations"],
            "future_work": paper["future_work"],
            "evidence": paper["evidence"]
        })


    prompt = f"""
You are a research gap analyst.

Research topic:
{state["topic"]}

Paper evidence:

{gap_data}

Compare the papers and identify research gaps.

Look especially at:

- repeated limitations
- missing datasets
- missing evaluation
- weaknesses in existing methods
- unexplored research areas
- future work mentioned by multiple papers

IMPORTANT:

- Only identify gaps supported by the provided information.
- Do not invent information.
- For every gap, mention the supporting paper titles.
- Use the provided evidence to explain why the gap exists.
- If only one paper mentions something, do not claim that it is a repeated gap.

Return a clear research-gap report.
"""

    result = model.invoke(prompt)

    print("\nResearch gaps:")
    print(result.content)

    return {
        "research_gaps": result.content
    }