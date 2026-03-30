"""
Phase 2: Full Finance Agent (5 tools) — SOLUTION
==================================================
Terminal-based agent with all five read-only analytical tools.

Run:  python3 phase2_full.py   (macOS/Linux)
       python phase2_full.py    (Windows)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from config import get_llm, SYSTEM_PROMPT, invoke_agent
from tools import (
    get_transactions,
    categorize_expenses,
    compare_months,
    find_subscriptions,
    search_transactions,
)

# All five analytical tools
tools = [
    get_transactions,
    categorize_expenses,
    compare_months,
    find_subscriptions,
    search_transactions,
]

llm = get_llm()
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def main():
    print("=" * 60)
    print("  Personal Finance Agent  (Phase 2 — 5 tools)")
    print("  Type 'quit' to exit.")
    print("=" * 60)
    print("\nTry asking:")
    print("  - 'Am I spending more in February than January? Why?'")
    print("  - 'Do I have any subscriptions?'")
    print("  - 'Search for Uber transactions'")

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
