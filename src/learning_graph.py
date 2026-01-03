# src/learning_graph.py

import os
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph import MessagesState

from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# -----------------------------
# Helpers
# -----------------------------
AMBIGUOUS_WORDS = [
    "best", "top", "some", "good", "learn", "course", "places"
]

def is_ambiguous(text: str) -> bool:
    text = text.lower()
    return any(word in text for word in AMBIGUOUS_WORDS)

def clarification_question(query: str) -> str:
    return (
        "I want to make sure I understand your request correctly.\n\n"
        f"You asked: **{query}**\n\n"
        "Could you please clarify:\n"
        "- Your goal or use case?\n"
        "- Any constraints or preferences?\n"
    )

# -----------------------------
# Nodes
# -----------------------------
def clarify_user(
    state: MessagesState
) -> Command[Literal["gather_context", "__end__"]]:

    messages = state["messages"]

    last_msg = messages[-1]

    # 🛑 If user already answered a clarification, DO NOT ask again
    if len(messages) >= 2 and isinstance(messages[-2], AIMessage):
        return Command(goto="gather_context")

    if isinstance(last_msg, HumanMessage) and is_ambiguous(last_msg.content):
        return Command(
            goto=END,
            update={
                "messages": [
                    AIMessage(content=clarification_question(last_msg.content))
                ]
            }
        )

    return Command(goto="gather_context")


def gather_context(state: MessagesState):
    """
    Gather context using Tavily after clarification is resolved.
    """

    # Combine all human messages into a single refined query
    user_inputs = [
        m.content for m in state["messages"]
        if isinstance(m, HumanMessage)
    ]
    final_query = " ".join(user_inputs)

    response = tavily_client.search(
        query=final_query,
        max_results=5,
        search_depth="advanced"
    )

    # ✅ Tavily returns a dict, not a list
    results = response.get("results", [])

    if not results:
        return {
            "messages": [
                AIMessage(content="I couldn’t find relevant information.")
            ]
        }

    answer = "Here are some relevant places based on your clarification:\n\n"
    for r in results[:3]:
        answer += f"- **{r.get('title', 'Unknown')}**\n  {r.get('url', '')}\n"

    return {
        "messages": [AIMessage(content=answer)]
    }

# -----------------------------
# Graph
# -----------------------------
builder = StateGraph(MessagesState)

builder.add_node("clarify_user", clarify_user)
builder.add_node("gather_context", gather_context)

builder.add_edge(START, "clarify_user")
builder.add_edge("gather_context", END)

graph = builder.compile()

# -----------------------------
# Local test
# -----------------------------
if __name__ == "__main__":
    user_input = input("User: ")

    state = {
        "messages": [HumanMessage(content=user_input)]
    }

    result = graph.invoke(state)

    for msg in result["messages"]:
        print(f"{msg.type.upper()}: {msg.content}")
