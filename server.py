import os
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from main import app, get_active_groq_model


# =========================================================
# FASTAPI SERVER
# =========================================================

server = FastAPI(
    title="Stacks Autonomous Researcher API"
)


# =========================================================
# CORS
# =========================================================

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HEALTH CHECK
# =========================================================

@server.get("/api/health")
async def health_check():

    return {
        "status": "online",
        "boot_id": "stacks-backend-v1",
        "groq_key": bool(
            os.getenv("GROQ_API_KEY")
        ),
        "active_model": get_active_groq_model()
    }


# =========================================================
# CHAT / RESEARCH ENDPOINT
# =========================================================

@server.post("/api/chat")
async def chat_endpoint(request: Request):

    data = await request.json()

    user_message = data.get(
        "message",
        ""
    ).strip()


    async def event_generator() -> AsyncGenerator[
        dict,
        None
    ]:

        # -------------------------------------------------
        # EMPTY QUERY
        # -------------------------------------------------

        if not user_message:

            yield {
                "event": "error",
                "data": json.dumps({
                    "message": "Empty query provided."
                })
            }

            return


        # -------------------------------------------------
        # INITIAL STATE
        # -------------------------------------------------

        initial_state = {

            "topic": user_message,

            "arxiv_papers": [],

            "open_alex": [],

            "relevant_papers": [],

            "paper_analysis": [],

            "novelty_assessments": []
        }


        # -------------------------------------------------
        # INITIAL UI MESSAGE
        # -------------------------------------------------

        yield {
            "event": "step",
            "data": json.dumps({

                "id": "step-init",

                "text":
                    f"Searching literature for: "
                    f"'{user_message}'",

                "status": "running"
            })
        }


        final_state = initial_state.copy()


        try:

            # =================================================
            # STREAM LANGGRAPH
            # =================================================

            async for output_chunk in app.astream(
                initial_state
            ):

                for node_name, node_state in output_chunk.items():

                    if not isinstance(
                        node_state,
                        dict
                    ):
                        continue


                    # Keep latest state
                    final_state.update(
                        node_state
                    )


                    # =================================================
                    # OPENALEX
                    # =================================================

                    if node_name == "openalex":

                        count = len(
                            node_state.get(
                                "open_alex",
                                []
                            )
                        )

                        yield {
                            "event": "step",
                            "data": json.dumps({

                                "id":
                                    "step-openalex",

                                "text":
                                    f"Found {count} papers "
                                    "on OpenAlex. "
                                    "Searching arXiv...",

                                "status":
                                    "running"
                            })
                        }


                    # =================================================
                    # ARXIV
                    # =================================================

                    elif node_name == "arxiv":

                        count = len(
                            node_state.get(
                                "arxiv_papers",
                                []
                            )
                        )

                        yield {
                            "event": "step",
                            "data": json.dumps({

                                "id":
                                    "step-arxiv",

                                "text":
                                    f"Found {count} papers "
                                    "on arXiv. "
                                    "Selecting relevant papers...",

                                "status":
                                    "running"
                            })
                        }


                    # =================================================
                    # RELEVANT PAPER FINDER
                    # =================================================

                    elif node_name == "relevant_paper_finder":

                        selected = node_state.get(
                            "relevant_papers",
                            []
                        )


                        # ---------------------------------------------
                        # SELECTION COMPLETE
                        # ---------------------------------------------

                        yield {
                            "event": "step",
                            "data": json.dumps({

                                "id":
                                    "step-finder",

                                "text":
                                    f"Selected "
                                    f"{len(selected)} "
                                    "candidate papers",

                                "status":
                                    "done"
                            })
                        }


                        # ---------------------------------------------
                        # IMPORTANT:
                        # Tell frontend what paper is being read
                        # BEFORE the paper_reader node finishes.
                        # ---------------------------------------------

                        if selected:

                            first_paper = selected[0]

                            first_title = first_paper.get(
                                "title",
                                "Untitled Paper"
                            )

                            yield {
                                "event": "step",
                                "data": json.dumps({

                                    "id":
                                        "step-reading",

                                    "text":
                                        f"Reading: "
                                        f"{first_title}",

                                    "status":
                                        "running"
                                })
                            }


                            # -----------------------------------------
                            # Send selected paper information to frontend
                            # -----------------------------------------

                            for idx, paper in enumerate(
                                selected,
                                start=1
                            ):

                                title = paper.get(
                                    "title",
                                    "Untitled Paper"
                                )

                                yield {
                                    "event": "paper",
                                    "data": json.dumps({

                                        "id":
                                            f"p-{idx}",

                                        "title":
                                            title,

                                        "pdf":
                                            paper.get(
                                                "pdf_url",
                                                ""
                                            ),

                                        "year":
                                            paper.get(
                                                "year",
                                                2026
                                            ),

                                        "source":
                                            paper.get(
                                                "source",
                                                "arxiv"
                                            ),

                                        "state":
                                            "selected"
                                    })
                                }


                    # =================================================
                    # PAPER READER
                    # =================================================

                    elif node_name == "paper_reader":

                        read_papers = node_state.get(
                            "relevant_papers",
                            []
                        )


                        # ---------------------------------------------
                        # Mark reading as completed
                        # ---------------------------------------------

                        if read_papers:

                            last_title = read_papers[-1].get(
                                "title",
                                "Untitled Paper"
                            )

                            yield {
                                "event": "step",
                                "data": json.dumps({

                                    "id":
                                        "step-reading",

                                    "text":
                                        f"Finished reading "
                                        f"{len(read_papers)} "
                                        "papers",

                                    "status":
                                        "done"
                                })
                            }


                        # ---------------------------------------------
                        # Update paper shelf
                        # ---------------------------------------------

                        for idx, paper in enumerate(
                            read_papers,
                            start=1
                        ):

                            title = paper.get(
                                "title",
                                "Untitled Paper"
                            )

                            yield {
                                "event": "paper",
                                "data": json.dumps({

                                    "id":
                                        f"p-{idx}",

                                    "title":
                                        title,

                                    "pdf":
                                        paper.get(
                                            "pdf_url",
                                            ""
                                        ),

                                    "year":
                                        paper.get(
                                            "year",
                                            2026
                                        ),

                                    "source":
                                        paper.get(
                                            "source",
                                            "arxiv"
                                        ),

                                    "state":
                                        "read"
                                })
                            }


                    # =================================================
                    # RESEARCH AGENT
                    # =================================================

                    elif node_name == "research_agent":

                        yield {
                            "event": "step",
                            "data": json.dumps({

                                "id":
                                    "step-agent",

                                "text":
                                    "Extracting research gaps, "
                                    "limitations, and methodologies...",

                                "status":
                                    "running"
                            })
                        }


                    # =================================================
                    # GAP ANALYZER
                    # =================================================

                    elif node_name == "gap_analyzer":

                        yield {
                            "event": "step",
                            "data": json.dumps({

                                "id":
                                    "step-gap",

                                "text":
                                    "Synthesizing novelty check "
                                    "& research gap assessments...",

                                "status":
                                    "done"
                            })
                        }


            # =================================================
            # FINAL MARKDOWN REPORT
            # =================================================

            assessments = final_state.get(
                "novelty_assessments",
                []
            )


            markdown_output = (
                "## EXISTING-WORK & "
                "RESEARCH-GAP VERIFICATION\n\n"
            )


            if not assessments:

                markdown_output += (
                    "No specific novelty assessments "
                    "were generated for this topic."
                )


            else:

                for idx, item in enumerate(
                    assessments,
                    start=1
                ):

                    markdown_output += (
                        f"### Topic "
                        f"{item.get('topic_number', idx)}: "
                        f"{item.get('topic_title', '')}\n\n"
                    )


                    # ---------------------------------------------
                    # STATUS
                    # ---------------------------------------------

                    markdown_output += (
                        "**Status:**\n"
                    )

                    markdown_output += (
                        f"`{item.get(
                            'status',
                            'Potentially underexplored'
                        )}`\n\n"
                    )


                    # ---------------------------------------------
                    # SIMILARITY
                    # ---------------------------------------------

                    markdown_output += (
                        "**Similarity Analysis:**\n"
                    )

                    markdown_output += (
                        f"{item.get(
                            'similarity_analysis',
                            ''
                        )}\n\n"
                    )


                    # ---------------------------------------------
                    # GAP
                    # ---------------------------------------------

                    markdown_output += (
                        "**Remaining Gap:**\n"
                    )

                    markdown_output += (
                        f"{item.get(
                            'remaining_gap',
                            ''
                        )}\n\n"
                    )


                    # ---------------------------------------------
                    # SEARCH COVERAGE
                    # ---------------------------------------------

                    coverage = item.get(
                        "search_coverage",
                        {}
                    )


                    if coverage:

                        markdown_output += (
                            "**Search Coverage:**\n"
                        )

                        markdown_output += (
                            f"- OpenAlex papers retrieved: "
                            f"{coverage.get(
                                'openalex_papers',
                                0
                            )}\n"
                        )

                        markdown_output += (
                            f"- arXiv papers retrieved: "
                            f"{coverage.get(
                                'arxiv_papers',
                                0
                            )}\n"
                        )

                        markdown_output += (
                            f"- Total papers retrieved: "
                            f"{coverage.get(
                                'total_retrieved',
                                0
                            )}\n"
                        )

                        markdown_output += (
                            f"- Unique papers with PDF URLs: "
                            f"{coverage.get(
                                'papers_with_pdf',
                                0
                            )}\n"
                        )

                        markdown_output += (
                            f"- Papers analyzed in depth: "
                            f"{coverage.get(
                                'papers_analyzed',
                                0
                            )}\n\n"
                        )


                    # ---------------------------------------------
                    # CONFIDENCE
                    # ---------------------------------------------

                    markdown_output += (
                        "**Evidence Confidence:** "
                    )

                    markdown_output += (
                        f"{item.get(
                            'evidence_confidence',
                            'Low'
                        )}\n\n"
                    )


                    # ---------------------------------------------
                    # EVIDENCE PAPERS
                    # ---------------------------------------------

                    markdown_output += (
                        "**Evidence Papers:**\n"
                    )


                    evidence_papers = item.get(
                        "evidence_papers",
                        []
                    )


                    if evidence_papers:

                        for paper in evidence_papers:

                            markdown_output += (
                                f"- {paper}\n"
                            )

                    else:

                        markdown_output += (
                            "- No specific evidence "
                            "papers returned.\n"
                        )


                    markdown_output += "\n"


                    # ---------------------------------------------
                    # COVERAGE LIMITATION
                    # ---------------------------------------------

                    markdown_output += (
                        "> *Coverage Limitation: "
                        f"{item.get(
                            'coverage_limitation',
                            'Assessment based on retrieved literature.'
                        )}*\n\n"
                    )


                    # ---------------------------------------------
                    # NOVELTY NOTE
                    # ---------------------------------------------

                    markdown_output += (
                        "> *Note: This assessment is based "
                        "on the retrieved literature and does "
                        "not establish definitive research "
                        "novelty.*\n\n"
                    )

                    markdown_output += "---\n\n"


            # =================================================
            # FINAL ANSWER
            # =================================================

            yield {
                "event": "answer",
                "data": json.dumps({
                    "text": markdown_output
                })
            }


            # =================================================
            # COMPLETED
            # =================================================

            yield {
                "event": "done",
                "data": json.dumps({
                    "status": "completed"
                })
            }


        except Exception as e:

            print(
                f"[SERVER ERROR] {str(e)}"
            )

            yield {
                "event": "error",
                "data": json.dumps({
                    "message":
                        f"Execution error: {str(e)}"
                })
            }


    return EventSourceResponse(
        event_generator()
    )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:server",
        host="127.0.0.1",
        port=8000,
        reload=True
    )