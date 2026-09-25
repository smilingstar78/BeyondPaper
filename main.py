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
    """Dynamically fetches the best available Groq model for your API key."""
    api_key = os.getenv("GROQ_API_KEY")
    fallback = "llama3-8b-8192" # Absolute fallback
    
    if not api_key:
        return fallback
        
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5
        )
        if response.status_code == 200:
            models = response.json().get("data", [])
            active_models = [m["id"] for m in models]
            
            print(f"\n[INFO] Found {len(active_models)} active models on your Groq tier.")
            
            preferred = [
                "llama-3.1-8b-instant",
                "llama-3.2-11b-vision-preview",
                "llama-3.2-3b-preview",
                "llama3-8b-8192",
                "gemma-7b-it"
            ]
            
            for pref in preferred:
                if pref in active_models:
                    print(f"[INFO] Auto-selected model: {pref}\n")
                    return pref
            
            if active_models:
                print(f"[INFO] Auto-selected fallback model: {active_models[0]}\n")
                return active_models[0]
                
    except Exception as e:
        print(f"\n[WARNING] Could not fetch models dynamically: {e}")
        
    return fallback


# =========================================================
# MCP CLIENT
# =========================================================

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
# LLM
# =========================================================

llm = ChatGroq(
    model=get_active_groq_model(),
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=3500,  # Set within the 4096 limit imposed by Groq API
    model_kwargs={"response_format": {"type": "json_object"}} 
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

    try:
        return json.loads(text)
    except Exception:
        pass

    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {}


# =========================================================
# CONVERT MCP RESULT
# =========================================================

def convert_tool_result(result):
    print("\nDEBUG MCP RESULT TYPE:", type(result))
    parsed_papers = []

    if isinstance(result, str):
        parsed = parse_json(result)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []

    if hasattr(result, "content"):
        return convert_tool_result(result.content)

    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and ("title" in item or "pdf_url" in item):
                parsed_papers.append(item)
            elif isinstance(item, dict) and "text" in item:
                parsed = parse_json(item["text"])
                if isinstance(parsed, list):
                    parsed_papers.extend(parsed)
                elif isinstance(parsed, dict):
                    parsed_papers.append(parsed)
            elif hasattr(item, "text"):
                parsed = parse_json(item.text)
                if isinstance(parsed, list):
                    parsed_papers.extend(parsed)
                elif isinstance(parsed, dict):
                    parsed_papers.append(parsed)
            elif isinstance(item, str):
                parsed = parse_json(item)
                if isinstance(parsed, list):
                    parsed_papers.extend(parsed)
                elif isinstance(parsed, dict):
                    parsed_papers.append(parsed)

        if parsed_papers:
            return parsed_papers
        return result

    if isinstance(result, dict):
        return [result]

    return []


# =========================================================
# OPENALEX NODE
# =========================================================

async def openalex_node(state):
    print("\nSearching OpenAlex...")
    tools = await client.get_tools()
    search_openalex = next((tool for tool in tools if tool.name == "search_openalex"), None)

    if search_openalex is None:
        raise ValueError("search_openalex tool was not found.")

    result = await search_openalex.ainvoke({"topic": state["topic"]})
    papers = convert_tool_result(result)

    if not isinstance(papers, list):
        papers = []

    print(f"OpenAlex found {len(papers)} papers.")
    return {"open_alex": papers}


# =========================================================
# ARXIV NODE
# =========================================================

async def arxiv_node(state):
    print("\nSearching arXiv...")
    tools = await client.get_tools()
    search_arxiv = next((tool for tool in tools if tool.name == "search_arxiv"), None)

    if search_arxiv is None:
        raise ValueError("search_arxiv tool was not found.")

    result = await search_arxiv.ainvoke({"topic": state["topic"]})
    papers = convert_tool_result(result)

    if not isinstance(papers, list):
        papers = []

    print(f"arXiv found {len(papers)} papers.")
    print("\nChecking arXiv PDF URLs...\n")

    for number, paper in enumerate(papers, start=1):
        print(f"{number}. {paper.get('title')}")
        print(f"   PDF: {paper.get('pdf_url')}")

    return {"arxiv_papers": papers}


# =========================================================
# FIND RELEVANT PAPERS
# =========================================================

async def relevant_paper_finder(state):
    print("\nFinding relevant papers...")
    openalex = state.get("open_alex", [])
    arxiv = state.get("arxiv_papers", [])
    all_papers = openalex + arxiv

    print(f"Total papers available: {len(all_papers)}")
    pdf_papers = []

    for paper in all_papers:
        if not isinstance(paper, dict):
            continue
        pdf_url = paper.get("pdf_url")
        if isinstance(pdf_url, str) and pdf_url.strip():
            pdf_papers.append(paper)

    print(f"Papers with PDF URLs: {len(pdf_papers)}")

    if not pdf_papers:
        print("\nNo papers with PDF URLs found.")
        return {"relevant_papers": []}

    paper_list = ""
    for index, paper in enumerate(pdf_papers):
        paper_list += f"{index}: {paper.get('title', 'Unknown')}\n"

    prompt = f"""
Select the 3 most relevant research papers for the given research topic.
Research topic: {state["topic"]}
Available papers:
{paper_list}

Return ONLY valid JSON.
Format:
{{
    "selected_indexes": [0, 1, 2]
}}
"""

    try:
        response = await llm.ainvoke(prompt)
        result = parse_json(response.content)
    except Exception as e:
        print(f"\nPaper selection failed: {e}")
        print("Using first 3 papers instead.")
        return {"relevant_papers": pdf_papers[:3]}

    indexes = result.get("selected_indexes", [])
    if not isinstance(indexes, list):
        indexes = []

    selected = []
    for index in indexes:
        try:
            index = int(index)
        except Exception:
            continue
        if 0 <= index < len(pdf_papers) and pdf_papers[index] not in selected:
            selected.append(pdf_papers[index])

    if not selected:
        print("\nLLM did not return valid indexes. Using first 3 papers.")
        selected = pdf_papers[:3]

    selected = selected[:3]

    print("\nSelected papers:")
    for number, paper in enumerate(selected, start=1):
        print(f"\n{number}. {paper.get('title')}")
        print(f"   PDF: {paper.get('pdf_url')}")

    return {"relevant_papers": selected}


# =========================================================
# PAPER READER NODE
# =========================================================

async def paper_reader_node(state):
    print("\nReading selected papers...")
    papers = state.get("relevant_papers", [])
    successful = []

    for paper in papers:
        title = paper.get("title", "Unknown")
        pdf_url = paper.get("pdf_url")

        if not pdf_url:
            print(f"\nSkipping: {title}\nReason: No PDF URL.")
            continue

        try:
            store_paper(title, pdf_url)
            successful.append(paper)
        except Exception as e:
            print(f"\nCould not read: {title}\nReason: {e}")

    print(f"\nSuccessfully stored {len(successful)} papers.")
    return {"relevant_papers": successful}


# =========================================================
# RESEARCH ANALYSIS NODE
# =========================================================

async def research_agent_node(state):
    print("\nAnalyzing papers...")
    papers = state.get("relevant_papers", [])
    all_analysis = []

    questions = {
        "research_problem": "What is the research problem, motivation, or research gap?",
        "objectives": "What are the objectives, goals, aims, or contributions?",
        "method": "What methodology, model, algorithm, or approach is proposed?",
        "dataset": "What datasets or data sources are used?",
        "results": "What are the main experimental results and findings?",
        "metrics": "What evaluation metrics are reported?",
        "limitations": "What limitations, weaknesses, or challenges are mentioned?",
        "future_work": "What future work or improvements are suggested?"
    }

    for paper in papers:
        title = paper.get("title", "Unknown")
        print(f"\nAnalyzing: {title}")
        evidence = {}

        for field, question in questions.items():
            try:
                chunks = search_paper(question, title, k=2)
                if chunks:
                    evidence[field] = "\n".join(chunks[:2])[:1200]
                else:
                    evidence[field] = "Not found."
            except Exception as e:
                evidence[field] = f"Retrieval error: {e}"

        evidence_text = ""
        for field, text in evidence.items():
            evidence_text += f"\n\n{field.upper()}:\n{str(text)[:1200]}"

        prompt = f"""
Analyze this research paper using ONLY the retrieved evidence.
Do not use outside knowledge. Do not invent information.

Keep your summaries concise (maximum 2 to 3 sentences per field) so the response does not cut off.

Paper Title: {title}

Retrieved evidence:
{evidence_text}

You must return ONLY a single valid JSON object. 
Extract the answers from the evidence and place them as strings in the corresponding JSON fields. 
If information is missing, write "Not found in retrieved evidence."

Format your output exactly like this:
{{
    "research_problem": "description of the problem...",
    "objectives": "description of the objectives...",
    "method": "description of the method...",
    "dataset": "description of the dataset...",
    "results": "description of the results...",
    "metrics": "description of the metrics...",
    "limitations": "description of the limitations...",
    "future_work": "description of the future work...",
    "evidence": "brief summary of evidence used..."
}}
"""

        try:
            response = await llm.ainvoke(prompt)
            analysis = parse_json(response.content)

            if not isinstance(analysis, dict) or not analysis:
                analysis = {"error": "Failed to parse JSON", "raw_response": response.content}

            analysis["paper"] = title
            all_analysis.append(analysis)

        except Exception as e:
            print(f"Analysis failed: {e}")
            all_analysis.append({"paper": title, "error": str(e)})

    return {"paper_analysis": all_analysis}


# =========================================================
# RESEARCH GAP & NOVELTY CHECK NODE
# =========================================================

async def gap_analyzer_node(state):
    print("\nEvaluating Literature & Generating Novelty Assessments...")
    topic = state.get("topic", "")
    analyses = state.get("paper_analysis", [])

    if not analyses:
        return {"novelty_assessments": []}

    summaries_text = ""
    evidence_papers = []
    for idx, item in enumerate(analyses, start=1):
        if "error" in item:
            continue
        paper_title = item.get('paper', 'Unknown')
        evidence_papers.append(paper_title)
        summaries_text += f"""
--- PAPER {idx}: {paper_title} ---
- Research Problem: {item.get('research_problem', 'N/A')}
- Method: {item.get('method', 'N/A')}
- Results: {item.get('results', 'N/A')}
- Limitations: {item.get('limitations', 'N/A')}
- Future Work: {item.get('future_work', 'N/A')}
"""

    prompt = f"""
You are an expert AI research advisor performing a Novelty and Existing-Work Analysis.

Target Research Topic Domain: {topic}

Extracted Analysis from Retrieved Literature:
{summaries_text}

Task:
Formulate 2 to 3 candidate research topics/titles based on open gaps in the retrieved literature.
For each topic, perform a similarity analysis against the retrieved evidence papers.

Return ONLY a valid JSON object formatted as follows:
{{
    "assessments": [
        {{
            "topic_number": 1,
            "topic_title": "Hierarchical Retrieval-Augmented Memory Networks for Long-Horizon Planning in Agentic AI",
            "status": "Potentially underexplored",
            "similarity_analysis": "Existing studies investigate RAG, memory, and long-horizon planning, but the retrieved literature does not clearly show a unified hierarchical memory architecture evaluated specifically for long-horizon agentic planning.",
            "remaining_gap": "Scalable integration of hierarchical memory with long-horizon planning remains insufficiently evaluated.",
            "evidence_confidence": "Medium",
            "evidence_papers": {json.dumps(evidence_papers)}
        }}
    ]
}}

Guidelines:
- "status": Choose from "Potentially underexplored", "Partially addressed", or "High overlap in literature".
- "evidence_confidence": Choose from "High", "Medium", or "Low".
- "similarity_analysis": Summarize how existing retrieved work relates and where it falls short.
- "remaining_gap": State the exact gap cleanly in 1-2 sentences.
"""

    try:
        response = await llm.ainvoke(prompt)
        res_json = parse_json(response.content)
        assessments = res_json.get("assessments", [])
        return {"novelty_assessments": assessments}

    except Exception as e:
        print(f"\n[ERROR] Novelty Assessment failed: {e}")
        return {"novelty_assessments": []}


# =========================================================
# GRAPH
# =========================================================

graph = StateGraph(MyState)
graph.add_node("openalex", openalex_node)
graph.add_node("arxiv", arxiv_node)
graph.add_node("relevant_paper_finder", relevant_paper_finder)
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
app = graph.compile()


# =========================================================
# MAIN
# =========================================================

async def main():
    topic = input("\nEnter research topic: ").strip()

    if not topic:
        print("Please enter a research topic.")
        return

    initial_state = {
        "topic": topic,
        "arxiv_papers": [],
        "open_alex": [],
        "relevant_papers": [],
        "paper_analysis": [],
        "novelty_assessments": []
    }

    final_state = await app.ainvoke(initial_state)

    assessments = final_state.get("novelty_assessments", [])

    print("\n" + "=" * 70)
    print("EXISTING-WORK / NOVELTY CHECK")
    print("=" * 70)

    if not assessments:
        print("\nNo novelty assessments generated.")
        return

    for item in assessments:
        print(f"\nTopic {item.get('topic_number', 1)}:")
        print(f"{item.get('topic_title', '')}\n")
        print("Status:")
        print(f"{item.get('status', 'Potentially underexplored')}\n")
        print("Similarity Analysis:")
        print(f"{item.get('similarity_analysis', '')}\n")
        print("Remaining Gap:")
        print(f"{item.get('remaining_gap', '')}\n")
        print("Evidence Confidence:")
        print(f"{item.get('evidence_confidence', 'Medium')}\n")
        print("Evidence Papers:")
        for paper in item.get("evidence_papers", []):
            print(f"- {paper}")
        print("\nNote:")
        print("This assessment is based on the retrieved literature and does not establish definitive novelty.")
        print("-" * 70)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())