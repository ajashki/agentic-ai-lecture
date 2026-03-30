"""
Phase 1: Basic Finance Agent (2 tools)
=======================================
Build a terminal-based agent that can retrieve transactions and categorize
expenses. You will wire up two pre-built tools and create a ReAct agent.

Run:  python3 phase1_basic.py   (macOS/Linux)
       python phase1_basic.py    (Windows)
"""

import sys
from pathlib import Path

# Allow imports from the workshop package
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# TODO (Phase 1, Step 1): Import the LLM helper
# Hint: from config import get_llm, SYSTEM_PROMPT
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TODO (Phase 1, Step 2): Import the two tools you need for this phase
# Hint: from tools import get_transactions, categorize_expenses
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TODO (Phase 1, Step 3): Create your tool list
# The tools are already decorated with @tool in finance_tools.py.
# Just put them in a plain Python list.
# Example: tools = [tool_a, tool_b]
# ---------------------------------------------------------------------------
tools = []  # <-- replace with your two tools

# ---------------------------------------------------------------------------
# TODO (Phase 1, Step 4): Create the LLM instance and build the agent
# 1. Call get_llm() to get a language model
# 2. Call create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
# Hint: from config import get_llm, SYSTEM_PROMPT
# ---------------------------------------------------------------------------
llm = None    # <-- replace with get_llm()
agent = None  # <-- replace with create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Chat loop (provided) — no changes needed here
# ---------------------------------------------------------------------------
from config import invoke_agent  # handles rate limits & errors gracefully


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
            continue  # error was already printed

        # The last message in the response is the agent's final answer
        ai_message = response["messages"][-1]
        print(f"\nAgent: {ai_message.content}")


if __name__ == "__main__":
    main()
