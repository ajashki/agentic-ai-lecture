"""
Phase 2: Full Finance Agent (5 tools)
======================================
Extend the agent with three more analytical tools so it can compare months,
find subscriptions, and search transactions by keyword.

Run:  python3 phase2_full.py   (macOS/Linux)
       python phase2_full.py    (Windows)
"""

import sys
from pathlib import Path

# Allow imports from the workshop package
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from config import get_llm, SYSTEM_PROMPT, invoke_agent
from tools import get_transactions, categorize_expenses

# ---------------------------------------------------------------------------
# TODO (Phase 2, Step 1): Import the three additional tools
# Hint: from tools import compare_months, find_subscriptions, search_transactions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TODO (Phase 2, Step 2): Create the full tool list with all 5 tools
# You already have get_transactions and categorize_expenses from Phase 1.
# Add compare_months, find_subscriptions, and search_transactions.
# ---------------------------------------------------------------------------
tools = [
    get_transactions,
    categorize_expenses,
    # <-- add the three new tools here
]

# Create the LLM and agent (same as Phase 1, but with more tools)
llm = get_llm()
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Chat loop (provided)
# ---------------------------------------------------------------------------
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
