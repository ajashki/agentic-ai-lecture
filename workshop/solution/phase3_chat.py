"""
Phase 3: Chainlit Chat UI with Memory & Human-in-the-Loop — SOLUTION
=====================================================================
Web-based chat agent with conversation memory and confirmation prompts
before executing action tools (set_budget_alert, calculate_savings_goal).

Run:  chainlit run phase3_chat.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chainlit as cl
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

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

# All 7 tools
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

# Memory keeps conversation history across messages in the same session
memory = MemorySaver()

# Build the agent — interrupt_before=["tools"] pauses before every tool call
# so we can check whether the tool needs user approval
agent = create_react_agent(
    get_llm(),
    tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=memory,
    interrupt_before=["tools"],
)

# Fixed thread ID so memory persists within a Chainlit session
THREAD_CONFIG = {"configurable": {"thread_id": "finance-chat"}}


@cl.on_message
async def handle_message(message: cl.Message):
    # 1. Invoke the agent with the user's message
    response = await ainvoke_agent(
        agent,
        {"messages": [HumanMessage(content=message.content)]},
        config=THREAD_CONFIG,
    )
    if response is None:
        await cl.Message(content="Something went wrong — check the terminal for details. Try again in a moment.").send()
        return

    # 2. Handle interrupts — the agent may pause before tool calls
    snapshot = await agent.aget_state(THREAD_CONFIG)
    while snapshot.next:
        last_msg = snapshot.values["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])
        needs_approval = any(tc["name"] in ACTION_TOOLS for tc in tool_calls)

        if needs_approval:
            tool_names = ", ".join(
                tc["name"] for tc in tool_calls if tc["name"] in ACTION_TOOLS
            )
            # Ask the user for confirmation
            res = await cl.AskActionMessage(
                content=f"The agent wants to use: **{tool_names}**. Allow?",
                actions=[
                    cl.Action(
                        name="approve",
                        payload={"approved": True},
                        label="Yes, go ahead",
                    ),
                    cl.Action(
                        name="deny",
                        payload={"approved": False},
                        label="No, skip it",
                    ),
                ],
            ).send()

            if res and res.get("payload", {}).get("approved"):
                # User approved — let the agent continue
                response = await ainvoke_agent(agent, None, config=THREAD_CONFIG)
            else:
                # User declined — send decline messages for action tools
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
            # Read-only tool — no approval needed, just continue
            response = await ainvoke_agent(agent, None, config=THREAD_CONFIG)

        if response is None:
            await cl.Message(content="Something went wrong — check the terminal for details. Try again in a moment.").send()
            return

        snapshot = await agent.aget_state(THREAD_CONFIG)

    # 3. Send the final response
    ai_message = response["messages"][-1]
    await cl.Message(content=ai_message.content).send()
