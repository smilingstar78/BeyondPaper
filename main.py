import os
import time
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_groq import ChatGroq

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tools


load_dotenv()


# -----------------------------
# Settings
# -----------------------------

MODEL_NAME = "openai/gpt-oss-120b"

# How many recent messages are sent to the LLM (keeps the context small).
# Keep this at 30 or more so one full research turn always fits.
MAX_HISTORY_MESSAGES = 40

# Max number of graph steps for a single question
RECURSION_LIMIT = 40


# -----------------------------
# System prompt
# -----------------------------

SYSTEM_PROMPT = """
You are an autonomous research assistant that the user chats with.

You have three tools:

1. search_arxiv      - search arXiv for papers (returns abstracts + PDF links)
2. search_open_alex  - search OpenAlex for papers (returns abstracts, citation counts, sometimes a PDF link)
3. read_paper        - read a paper from its PDF URL (returns the start and the end of the paper)

HOW TO WORK

- Decide yourself which tools are needed. There is no fixed order.
- For a new research topic: search arXiv and/or OpenAlex, pick the most relevant
  papers (prefer recent and highly cited ones), and read up to 3 of them with read_paper.
  Only read papers that have a PDF link.
- After every tool result ask yourself: "Do I have enough reliable evidence to
  answer?" If not, use another tool. If yes, stop and answer.
- If a tool returns an error, try another tool or another query instead of giving up.
- For follow-up questions, FIRST use the information already collected in this
  conversation. Only call tools again if you need new information (a new topic,
  a different paper, more evidence).

WHAT TO PRODUCE

When the user asks for a research overview, research gaps or new topics, answer with:

1. Overview - what the field is about, in a few sentences.
2. Key papers - title, year, and one line on what each contributes.
3. Research gaps - specific, evidence-based gaps. For each gap say which
   paper(s) it comes from (limitations, future work, missing evaluation, etc.).
4. Suggested research topics - concrete, novel directions that follow from the gaps.

For other questions, just answer the question directly and clearly.

RULES

- Base your answer only on the papers you actually found. Never invent papers,
  authors, numbers or citations.
- Be honest about how much you used: say whether a claim comes from a full read
  of the paper or only from its abstract.
- Clearly separate what the papers state from your own suggestions.
- Do not explain your hidden reasoning. Give only the useful result.
"""


# -----------------------------
# State
# -----------------------------

class MyState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# Graph
# -----------------------------

def build_graph(llm):
    """Build and compile the research agent graph for the given chat model."""

    llm_with_tools = llm.bind_tools(tools)

    def research_agent(state: MyState):

        # Send only the most recent messages (always starts on a human
        # message so tool calls and tool results are never separated).
        history = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,
            max_tokens=MAX_HISTORY_MESSAGES,
            start_on="human",
            include_system=False,
        )

        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history

        last_error = None

        # Retry a few times (Groq sometimes returns a malformed tool call)
        for attempt in range(3):
            try:
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                last_error = e
                time.sleep(2 * (attempt + 1))

        raise last_error

    graph = StateGraph(MyState)

    graph.add_node("research_agent", research_agent)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "research_agent")

    # research_agent -> tools (if the LLM called a tool) OR -> END
    graph.add_conditional_edges("research_agent", tools_condition)

    # tools -> research_agent
    graph.add_edge("tools", "research_agent")

    # MemorySaver keeps the conversation, so follow-up questions work
    return graph.compile(checkpointer=MemorySaver())


# -----------------------------
# Chat helpers
# -----------------------------

def _text(content) -> str:
    """Message content can be a string or a list of blocks."""
    if isinstance(content, str):
        return content

    parts = []

    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))

    return "".join(parts)


def ask(chatbot, question: str, thread_id: str) -> str:
    """Send one user message, show tool activity live, return the final answer."""

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    final_answer = ""

    for chunk in chatbot.stream(
        {"messages": [("user", question)]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():

            if not update:
                continue

            for message in update.get("messages", []):

                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        for call in message.tool_calls:
                            print(f"  -> using {call['name']}({call['args']})")
                    else:
                        final_answer = _text(message.content)

                elif isinstance(message, ToolMessage):
                    print(f"  <- {message.name} finished")

    return final_answer


# -----------------------------
# Chat loop
# -----------------------------

def main():

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is missing. Add it to your .env file.")

    if not os.getenv("OPENALEX_API_KEY"):
        print(
            "Note: OPENALEX_API_KEY is not set. OpenAlex now needs a free key "
            "(https://openalex.org/settings/api). arXiv will still work.\n"
        )

    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    chatbot = build_graph(llm)

    thread_id = str(uuid.uuid4())

    print("=" * 60)
    print("RESEARCH AGENT")
    print("=" * 60)
    print("Ask me to research a topic, find research gaps, or suggest topics.")
    print("Then ask follow-up questions about the papers.")
    print("Commands:  /new = start a new conversation   exit = quit\n")

    while True:

        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Bye!")
            break

        if question.lower() == "/new":
            thread_id = str(uuid.uuid4())
            print("Started a new conversation.\n")
            continue

        try:
            answer = ask(chatbot, question, thread_id)
        except KeyboardInterrupt:
            print("\nStopped.\n")
            continue
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        print("\n" + "=" * 60)
        print("AGENT")
        print("=" * 60)
        print(answer or "(no answer returned)")
        print()


if __name__ == "__main__":
    main()