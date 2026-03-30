# Personal Finance Agent -- Student Guide

## Workshop Overview

In this workshop you will build a **Personal Finance Agent** -- an AI system that can reason about your spending, choose which tools to call, and adapt its investigation based on what it finds.

You will progress through three phases:

1. **Terminal agent** with two basic tools (retrieve transactions, categorize expenses).
2. **Full tool suite** where the agent chooses and chains five tools to answer complex questions.
3. **Chat UI** powered by Chainlit, with conversation memory and human-in-the-loop confirmation before the agent takes actions.

**Why is this agentic?** A traditional program would need hard-coded logic for every possible question. Your agent *decides* which tools to call, interprets intermediate results, and plans its next step -- the investigation path is emergent, not pre-programmed.

**Tech stack**: Python 3.10+, LangGraph, Gemini (or Ollama fallback), Chainlit.

---

## Phase 1: Basic Agent (30 min)

### What you will build

A terminal-based ReAct agent with two tools:

| Tool | Purpose |
|------|---------|
| `get_transactions(month)` | Returns raw transaction data for a given month |
| `categorize_expenses(month)` | Returns a dictionary of spending totals by category |

### Step-by-step instructions

Open `starter/phase1_basic.py`. You will fill in four TODO blocks.

**Step 1 -- Import the LLM helper**

```python
from config import get_llm, SYSTEM_PROMPT
```

**Step 2 -- Import the two tools**

```python
from tools import get_transactions, categorize_expenses
```

**Step 3 -- Create your tool list**

```python
tools = [get_transactions, categorize_expenses]
```

**Step 4 -- Create the LLM and build the agent**

```python
llm = get_llm()
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
```

**Run it:**

**macOS / Linux:**
```bash
python3 starter/phase1_basic.py
```

**Windows:**
```cmd
python starter/phase1_basic.py
```

### Test queries to try

- "How much did I spend in January?"
- "What did I spend the most on in January?"
- "Show me all my January transactions."
- "Which category had the highest spending in February?"

### Checkpoint

You should see:

- The agent reasoning about *which* tool to call (e.g., choosing `categorize_expenses` for a "how much" question vs `get_transactions` for a "show me" question).
- Tool calls appearing in the output.
- A coherent natural-language answer summarizing the results.

If "How much did I spend in January?" triggers `categorize_expenses` and returns a breakdown by category with a total, you are on track.

---

## Phase 2: Full Tool Suite (35 min)

### What you will add

Three more tools that give the agent richer investigative capabilities:

| Tool | Purpose |
|------|---------|
| `compare_months(month1, month2)` | Returns a comparison dict with deltas between two months |
| `find_subscriptions()` | Detects recurring charges across your transaction history |
| `search_transactions(keyword)` | Searches transactions by merchant name or description |

### Step-by-step instructions

Open `starter/phase2_full.py`. Phase 1 code is already filled in for you. You will add the new tools.

**Step 1 -- Import the new tools**

Add `compare_months`, `find_subscriptions`, and `search_transactions` to your import from `tools`.

**Step 2 -- Add them to the tool list**

```python
tools = [get_transactions, categorize_expenses, compare_months, find_subscriptions, search_transactions]
```

**Step 3 -- Rebuild the agent with all 5 tools**

The agent is already created for you in the starter code (with `SYSTEM_PROMPT`). Just make sure your tool list is complete.

**Run it:**

**macOS / Linux:**
```bash
python3 starter/phase2_full.py
```

**Windows:**
```cmd
python starter/phase2_full.py
```

### Test queries to try

- "Am I spending more in February than January? Why?"
- "Find any subscriptions I might have forgotten about."
- "How much am I spending on Uber Eats?"
- "I feel like I'm spending too much on food, is that true?"
- "What changed the most between January and March?"

### Checkpoint

You should see the agent **chaining multiple tools**. For example, when asked "Am I spending more this month than last?":

1. Agent calls `compare_months` to get the delta.
2. Spots that food spending increased.
3. Calls `search_transactions("food")` or `get_transactions` to drill deeper.
4. Explains the pattern in its answer.

The key insight: the agent uses the *result* of one tool to decide what to investigate next.

---

## Phase 3: Chat UI + Memory + Human-in-the-Loop (30 min)

### What you will add

- **Chainlit chat UI** -- a real web-based chat interface instead of the terminal.
- **MemorySaver** -- the agent remembers previous messages in the conversation.
- **Human-in-the-loop** -- the agent pauses and asks for your confirmation before taking certain actions (like setting a budget alert).

### Step-by-step instructions

Open `starter/phase3_chat.py`. The tools and Chainlit handler are already provided. You have 3 TODOs to fill in.

**Step 1 -- Import MemorySaver**

```python
from langgraph.checkpoint.memory import MemorySaver
```

**Step 2 -- Create a checkpointer**

```python
memory = MemorySaver()
```

This stores conversation history in memory so the agent remembers earlier messages.

**Step 3 -- Create the agent with memory and interrupt_before**

```python
agent = create_react_agent(
    get_llm(), tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=memory,
    interrupt_before=["tools"],
)
```

`interrupt_before=["tools"]` makes the agent pause before every tool call. The Chainlit handler (already provided below the TODOs) checks whether the tool needs approval and either auto-continues or asks you to confirm.

**Run it:**

```bash
chainlit run starter/phase3_chat.py -w
```

Then open `http://localhost:8000` in your browser.

### Test scenarios

- **Memory**: Ask "How much did I spend in January?" then follow up with "What about February?" -- the agent should understand you are still asking about spending.
- **Multi-turn**: Have a back-and-forth conversation drilling into your finances.
- **Human-in-the-loop**: Ask "Set a budget alert for food at $200." The agent should propose the alert and wait for your confirmation before executing `set_budget_alert`.

### Checkpoint

You should have a full chat application running in your browser:

- ChatGPT-like interface with message bubbles.
- The agent remembers context across messages.
- The agent asks for confirmation before setting budget alerts.
- You could share your screen and it looks like a real product.

---

## Bonus: Multi-Agent Extension

If you finish early, try splitting your single agent into two specialized agents:

- **Analyst Agent** -- has the read-only tools: `get_transactions`, `categorize_expenses`, `compare_months`, `find_subscriptions`, `search_transactions`.
- **Advisor Agent** -- has the action tools: `calculate_savings_goal`, `set_budget_alert`.

Build a handoff flow using LangGraph's `StateGraph`:

1. The Analyst investigates the user's finances.
2. It hands its findings to the Advisor.
3. The Advisor recommends actions based on the analysis.

This demonstrates why you would split into multiple agents: separation of concerns, principle of least privilege (the analyst cannot modify anything), and clearer reasoning within each agent's domain.

See `solution/bonus_multi_agent.py` for a reference implementation.

---

## Debugging Tips

| Problem | Cause | Fix |
|---------|-------|-----|
| `429 Resource Exhausted` | Gemini free-tier rate limit (10 RPM) | Wait 60 seconds and try again. If persistent, switch to Ollama. |
| `ImportError: No module named 'langgraph'` | Packages not installed | Run `pip3 install -r requirements.txt` again |
| `Invalid API key` or `GOOGLE_API_KEY not set` | Missing or incorrect API key | Check your `.env` file. Run `python3 check_setup.py` (macOS/Linux) or `python check_setup.py` (Windows) to verify. |
| Agent loops forever / repeats tool calls | Tool returning unexpected or empty values | Check that your tool functions return meaningful data. Print the return value to debug. |
| `ModuleNotFoundError` for tools | `sys.path` not set up correctly | Make sure `sys.path.insert(0, str(Path(__file__).parent.parent))` is at the top of your file. |
| Chainlit won't start | Port 8000 already in use | Kill the existing process or use a different port: `chainlit run phase3_chat.py -w --port 8001` |
| Chainlit shows blank page | Browser cache or WebSocket issue | Hard-refresh the page (Ctrl+Shift+R) or try a different browser. |
| `TypeError: 'NoneType' object is not callable` | Forgot to replace `None` placeholders in the starter code | Go back and fill in the TODO blocks -- `llm` and `agent` must not be `None`. |
| Agent does not use a tool you added | Tool missing from the tools list, or docstring is missing | Verify the tool is in your `tools = [...]` list. Each `@tool` function needs a docstring -- the agent reads it to know when to use the tool. |
| Ollama: `connection refused` | Ollama server not running | Start it first: `ollama serve` (keep that terminal open) |

## Using Ollama instead of Gemini

If you don't have a Gemini API key or it's not working, the code automatically falls back to Ollama.

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull a model: `ollama pull qwen2.5:3b`
3. Start the server: `ollama serve` (keep this terminal open)
4. In your `.env` file, leave the API key empty or comment it out:
   ```
   # GOOGLE_API_KEY=
   ```
5. Run `python3 check_setup.py` (macOS/Linux) or `python check_setup.py` (Windows) to verify Ollama works
