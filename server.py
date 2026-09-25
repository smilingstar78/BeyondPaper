
from fastapi import FastAPI
from pydantic import BaseModel

from main import app as research_graph


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="BeyondPaper API"
)


# =========================================================
# REQUEST
# =========================================================

class ResearchRequest(BaseModel):

    topic: str


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
async def root():

    return {
        "status": "ok",
        "message": "BeyondPaper API is running."
    }


# =========================================================
# RESEARCH
# =========================================================

@app.post("/research")
async def research(
    request: ResearchRequest
):

    topic = request.topic.strip()

    if not topic:

        return {
            "error": "Research topic is required."
        }

    initial_state = {

        "topic": topic,

        "arxiv_papers": [],

        "open_alex": [],

        "relevant_papers": [],

        "paper_analysis": [],

        "novelty_assessments": []
    }

    final_state = await research_graph.ainvoke(
        initial_state
    )

    return {
        "topic": topic,

        "assessments": final_state.get(
            "novelty_assessments",
            []
        )
    }
