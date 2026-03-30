"""
Phase 3: Chainlit Chat UI with Memory & Human-in-the-Loop
==========================================================
Wrap the agent in a Chainlit web interface. Add conversation memory so the
agent remembers context, and add a human-in-the-loop confirmation step
before the agent can set budgets or calculate savings goals.

Run:  chainlit run phase3_chat.py
"""

import sys
from pathlib import Path

# Allow imports from the workshop package
sys.path.insert(0, str(Path(__file__).parent.parent))

import chainlit as cl
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# TODO (Phase 3, Step 1): Import MemorySaver for conversation persistence
# Hint: from langgraph.checkpoint.memory import MemorySaver
# ---------------------------------------------------------------------------


from config import get_llm, SYSTEM_PROMPT, ainvoke_agent
from tools import (
    get_transactions,
    categorize_expenses,
    compare_months,
    find_subscriptions,
    search_transactions,
    calculate_savings_goal,
    set_budget_alert,
)

# All 7 tools — the read-only tools plus two action tools
tools = [
    get_transactions,
    categorize_expenses,
    compare_months,
    find_subscriptions,
    search_transactions,
    calculate_savings_goal,
    set_budget_alert,
]

# Tools that require human confirmation before executing
ACTION_TOOLS = {"calculate_savings_goal", "set_budget_alert"}

# ---------------------------------------------------------------------------
# TODO (Phase 3, Step 2): Create a MemorySaver checkpointer
# Hint: memory = MemorySaver()
# ---------------------------------------------------------------------------
memory = None  # <-- replace with MemorySaver()

# ---------------------------------------------------------------------------
# TODO (Phase 3, Step 3): Create the agent with interrupt_before
# When the agent wants to call an action tool, it should pause and ask the
# user for confirmation first.
# Hint: agent = create_react_agent(
#     get_llm(), tools,
#     prompt=SYSTEM_PROMPT,
#     checkpointer=memory,
#     interrupt_before=["tools"],
# )
# ---------------------------------------------------------------------------
agent = None  # <-- replace with create_react_agent(...)

# We use a fixed thread ID so memory persists within a single Chainlit session
THREAD_CONFIG = {"configurable": {"thread_id": "finance-chat"}}


# ---------------------------------------------------------------------------
# Chainlit message handler (provided) — read through to understand the flow
#
# How it works:
#   1. Every user message triggers handle_message via @cl.on_message.
#   2. The agent runs but pauses before each tool call (interrupt_before).
#   3. If the tool is an ACTION_TOOL, we ask the user for confirmation.
#   4. If it's a read-only tool, we let it proceed automatically.
#   5. The final response is sent back to the chat UI.
# ---------------------------------------------------------------------------

@cl.on_message
async def handle_message(message: cl.Message):
    # 1. Invoke the agent with the user's message
    response = await ainvoke_agent(
        agent,
        {"messages": [HumanMessage(content=message.content)]},
        config=THREAD_CONFIG,
    )
    if response is None:
        await cl.Message(content="Something went wrong — check the terminal for details.").send()
        return

    # 2. Check for interrupted tool calls that need confirmation
    snapshot = await agent.aget_state(THREAD_CONFIG)
    while snapshot.next:  # The agent is paused before a tool call
        # Find which tool the agent wants to call
        last_msg = snapshot.values["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])
        needs_approval = any(tc["name"] in ACTION_TOOLS for tc in tool_calls)

        if needs_approval:
            tool_names = ", ".join(tc["name"] for tc in tool_calls if tc["name"] in ACTION_TOOLS)
            # Ask the user for confirmation
            res = await cl.AskActionMessage(
                content=f"The agent wants to use: **{tool_names}**. Allow?",
                actions=[
                    cl.Action(name="approve", payload={"approved": True}, label="Yes, go ahead"),
                    cl.Action(name="deny", payload={"approved": False}, label="No, skip it"),
                ],
            ).send()

            if res and res.get("payload", {}).get("approved"):
                response = await ainvoke_agent(agent, None, config=THREAD_CONFIG)
            else:
                # Tell the agent the user said no
                from langchain_core.messages import ToolMessage
                decline_msgs = []
                for tc in tool_calls:
                    if tc["name"] in ACTION_TOOLS:
                        decline_msgs.append(
                            ToolMessage(
                                content="User declined this action.",
                                tool_call_id=tc["id"],
                            )
                        )
                await agent.aupdate_state(THREAD_CONFIG, {"messages": decline_msgs})
                response = await ainvoke_agent(agent, None, config=THREAD_CONFIG)
        else:
            # Tool doesn't need approval — just continue
            response = await ainvoke_agent(agent, None, config=THREAD_CONFIG)

        if response is None:
            await cl.Message(content="Something went wrong — check the terminal for details.").send()
            return

        snapshot = await agent.aget_state(THREAD_CONFIG)

    # 3. Send the final response
    ai_message = response["messages"][-1]
    await cl.Message(content=ai_message.content).send()
