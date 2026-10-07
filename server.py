import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from main import research_graph


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="BeyondPaper API"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODEL
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
# NODE LABELS
# =========================================================

NODE_LABELS = {
    "openalex": "Searching OpenAlex...",
    "arxiv": "Searching arXiv...",
    "relevant_paper_finder": "Selecting the most relevant papers...",
    "paper_reader": "Reading and indexing papers...",
    "research_agent": "Analyzing papers for problems, methods, and gaps...",
    "gap_analyzer": "Identifying research gaps and directions..."
}


# =========================================================
# SSE EVENT
# =========================================================

def sse_event(event_type: str, data: dict) -> str:

    payload = json.dumps(
        {
            "type": event_type,
            **data
        },
        ensure_ascii=False
    )

    return f"data: {payload}\n\n"


# =========================================================
# RESEARCH STREAM
# =========================================================

async def run_research_stream(
    topic: str
) -> AsyncGenerator[str, None]:

    initial_state = {
        "topic": topic,
        "arxiv_papers": [],
        "open_alex": [],
        "relevant_papers": [],
        "paper_analysis": [],
        "novelty_assessments": []
    }

    final_state = dict(initial_state)

    try:

        # -------------------------------------------------
        # RUN GRAPH ONCE
        # -------------------------------------------------

        async for chunk in research_graph.astream(
            initial_state,
            stream_mode="updates"
        ):

            if not chunk:
                continue

            for node_name, node_output in chunk.items():

                if isinstance(node_output, dict):

                    final_state.update(
                        node_output
                    )

                yield sse_event(
                    "progress",
                    {
                        "node": node_name,
                        "message": NODE_LABELS.get(
                            node_name,
                            f"Running {node_name}..."
                        )
                    }
                )

        # -------------------------------------------------
        # FINAL ANSWER
        # -------------------------------------------------

        yield sse_event(
            "answer",
            {
                "topic": topic,
                "assessments": final_state.get(
                    "novelty_assessments",
                    []
                )
            }
        )

        # -------------------------------------------------
        # DONE
        # -------------------------------------------------

        yield sse_event(
            "done",
            {
                "status": "completed"
            }
        )

    except Exception as error:

        print(
            f"\n[SERVER ERROR] {type(error).__name__}: {error}"
        )

        yield sse_event(
            "error",
            {
                "message": (
                    f"{type(error).__name__}: {error}"
                )
            }
        )


# =========================================================
# RESEARCH ENDPOINT
# =========================================================

@app.post("/research")
async def research(
    request: ResearchRequest
):

    topic = request.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Research topic is required."
        )

    return StreamingResponse(
        run_research_stream(topic),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =========================================================
# RUN LOCALLY
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
