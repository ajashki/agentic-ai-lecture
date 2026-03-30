"""
Phase 1: Basic Finance Agent (2 tools) — SOLUTION
===================================================
Terminal-based agent that retrieves transactions and categorizes expenses.

Run:  python3 phase1_basic.py   (macOS/Linux)
       python phase1_basic.py    (Windows)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from config import get_llm, SYSTEM_PROMPT, invoke_agent
from tools import get_transactions, categorize_expenses

# Two tools for Phase 1
tools = [get_transactions, categorize_expenses]

# Create the LLM and build the agent
llm = get_llm()
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def main():
    print("=" * 60)
    print("  Personal Finance Agent  (Phase 1 — 2 tools)")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = invoke_agent(agent, {"messages": [HumanMessage(content=user_input)]})
        if response is None:
            continue
        ai_message = response["messages"][-1]
        print(f"\nAgent: {ai_message.content}")


if __name__ == "__main__":
    main()
