import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

    return {
        "status": "ok",
        "message": "BeyondPaper API is running."
    }


# =========================================================
# STREAMING HELPERS
# =========================================================

# Human-readable labels shown to the user as each graph node runs.
NODE_LABELS = {
    "openalex": "Searching OpenAlex...",
    "arxiv": "Searching arXiv...",
    "relevant_paper_finder": "Selecting the most relevant papers...",
    "paper_reader": "Reading and indexing papers...",
    "research_agent": "Analyzing papers for problems, methods, and gaps...",
    "gap_analyzer": "Identifying research gaps and directions...",
}


def sse_event(event_type: str, data: dict) -> str:
    """
    Format a single Server-Sent Event line. Each event is a JSON
    payload with a "type" field the frontend switches on
    (progress / answer / error / done).
    """

    payload = json.dumps(
        {"type": event_type, **data},
        ensure_ascii=False
    )

    return f"data: {payload}\n\n"


async def run_research_stream(topic: str):
    """
    Runs the LangGraph pipeline via astream() so we can emit a
    progress event after each node finishes, instead of blocking
    silently until the whole multi-minute pipeline completes.
    """

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

        # stream_mode="updates" yields {node_name: partial_state}
        # after each node completes, so we know exactly which step
        # just ran and can label it for the user.
        async for chunk in research_graph.astream(
            initial_state,
            stream_mode="updates"
        ):

            for node_name, node_output in chunk.items():

                final_state.update(node_output)

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

    except Exception as error:

        yield sse_event(
            "error",
            {"message": f"Research pipeline failed: {error}"}
        )

        return

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

    yield sse_event(
        "done",
        {"status": "completed"}
    )


# =========================================================
# ENDPOINT
# =========================================================

@app.post("/research")
async def research(
    request: ResearchRequest
):

    topic = request.topic.strip()

    if not topic:
        # FIX: raise a proper HTTP error instead of returning
        # {"error": ...} with a 200 status. The frontend only
        # treats a response as an error when res.ok is false,
        # so a 200 response was silently swallowed before.
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
            # Disables buffering on some proxies (e.g. nginx) so
            # events reach the client as they're produced.
            "X-Accel-Buffering": "no",
        }
    )
