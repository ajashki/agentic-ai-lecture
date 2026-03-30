"""
Bonus: Multi-Agent Finance System
===================================
Two specialised agents that collaborate via a LangGraph StateGraph:

  - Analyst Agent  (read-only tools) — investigates transactions, finds
    patterns, compares months, and identifies subscriptions.
  - Advisor Agent  (action tools) — recommends savings goals and sets
    budget alerts based on the analyst's findings.

The flow is simple: Analyst investigates first, then Advisor recommends.

Run:  python3 bonus_multi_agent.py   (macOS/Linux)
       python bonus_multi_agent.py    (Windows)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from config import get_llm, invoke_agent
from tools import (
    get_transactions,
    categorize_expenses,
    compare_months,
    find_subscriptions,
    search_transactions,
    calculate_savings_goal,
    set_budget_alert,
)

# ── Tool groups ──────────────────────────────────────────────────────────────

analyst_tools = [
    get_transactions,
    categorize_expenses,
    compare_months,
    find_subscriptions,
    search_transactions,
]

advisor_tools = [
    calculate_savings_goal,
    set_budget_alert,
]

# ── Build individual agents ──────────────────────────────────────────────────

llm = get_llm()

analyst_agent = create_react_agent(
    llm,
    analyst_tools,
    prompt=SystemMessage(
        content=(
            "You are a financial analyst agent. Today's date is March 15, 2025. "
            "You have access to transaction data for January, February, and March 2025. "
            "Your job is to investigate the user's transactions and spending patterns. "
            "Use your tools to gather data, then summarize your findings clearly. "
            "Do NOT give financial advice — just report the facts. End your response "
            "with a section called 'FINDINGS SUMMARY' that lists key observations."
        )
    ),
)

advisor_agent = create_react_agent(
    llm,
    advisor_tools,
    prompt=SystemMessage(
        content=(
            "You are a financial advisor agent. You receive an analyst's findings "
            "about a user's spending. Based on those findings, use your tools to "
            "calculate savings goals and set budget alerts where appropriate. "
            "Give clear, actionable recommendations."
        )
    ),
)


# ── Multi-agent graph state ─────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    analyst_summary: str


# ── Graph nodes ──────────────────────────────────────────────────────────────

def analyst_node(state: AgentState) -> dict:
    """Run the analyst agent to investigate the user's query."""
    response = invoke_agent(analyst_agent, {"messages": state["messages"]})
    if response is None:
        return {"messages": [AIMessage(content="[Analyst]\nError — could not complete analysis.")], "analyst_summary": "Error"}
    # Extract the final AI message as the summary
    ai_messages = [m for m in response["messages"] if isinstance(m, AIMessage)]
    summary = ai_messages[-1].content if ai_messages else "No findings."
    return {
        "messages": [AIMessage(content=f"[Analyst]\n{summary}")],
        "analyst_summary": summary,
    }


def advisor_node(state: AgentState) -> dict:
    """Run the advisor agent, providing the analyst's findings as context."""
    # Build a message sequence for the advisor
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    original_query = user_msgs[0].content if user_msgs else ""

    advisor_input = [
        HumanMessage(
            content=(
                f"The user asked: {original_query}\n\n"
                f"Here are the analyst's findings:\n{state['analyst_summary']}\n\n"
                "Based on these findings, please provide recommendations. "
                "Use your tools (calculate_savings_goal, set_budget_alert) "
                "where it makes sense."
            )
        )
    ]
    response = invoke_agent(advisor_agent, {"messages": advisor_input})
    if response is None:
        return {"messages": [AIMessage(content="[Advisor]\nError — could not generate recommendations.")]}
    ai_messages = [m for m in response["messages"] if isinstance(m, AIMessage)]
    advice = ai_messages[-1].content if ai_messages else "No recommendations."
    return {
        "messages": [AIMessage(content=f"[Advisor]\n{advice}")],
    }


# ── Build the graph ─────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)
workflow.add_node("analyst", analyst_node)
workflow.add_node("advisor", advisor_node)

workflow.add_edge(START, "analyst")
workflow.add_edge("analyst", "advisor")
workflow.add_edge("advisor", END)

graph = workflow.compile()


# ── Terminal chat loop ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Multi-Agent Finance System  (Analyst + Advisor)")
    print("  Type 'quit' to exit.")
    print("=" * 60)
    print("\nThe Analyst will investigate your query first,")
    print("then the Advisor will give recommendations.\n")
    print("Try: 'Analyze my spending and help me save 500 euros in 6 months'")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\n--- Analyst is investigating... ---\n")
        result = graph.invoke({
            "messages": [HumanMessage(content=user_input)],
            "analyst_summary": "",
        })

        # Print the last two messages (analyst + advisor)
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content.startswith("["):
                print(msg.content)
                print()


if __name__ == "__main__":
    main()
