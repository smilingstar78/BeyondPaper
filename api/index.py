import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import research_graph


app = FastAPI(
    title="BeyondPaper API"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str


@app.get("/")
async def root():

    groq_key = os.environ.get(
        "GROQ_API_KEY"
    )

    return {
        "status": "ok",
        "message": "BeyondPaper API is running.",
        "groq_key": bool(groq_key)
    }


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
