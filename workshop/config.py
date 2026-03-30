import asyncio
import os
import time
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

load_dotenv()

# ANSI colors for terminal output
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"

# System prompt gives the agent date context so it can resolve "this month" etc.
SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful personal finance assistant. "
    "Today's date is March 15, 2025. "
    "You have access to transaction data for January, February, and March 2025."
))


def get_llm():
    """Return a Gemini LLM if an API key is available, otherwise fall back to Ollama."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Default: gemini-2.5-flash-lite — 30 RPM free tier, good for workshops
        # Alternative: gemini-2.5-flash — smarter but only 10 RPM free tier
        # Full model list: https://ai.google.dev/gemini-api/docs/models
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
        )
    else:
        from langchain_ollama import ChatOllama

        # Default: qwen2.5:3b — good tool calling, works on 8GB RAM (~3.5GB)
        # Alternative: llama3.1:8b — best quality, needs 16GB+ RAM (~6-7GB runtime)
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        print(f"{_YELLOW}No GOOGLE_API_KEY found in .env — falling back to Ollama ({model}).{_RESET}")
        print("Make sure Ollama is running: ollama serve")
        return ChatOllama(model=model)


def invoke_agent(agent, inputs, config=None, max_retries=3):
    """Invoke an agent with automatic retry on rate-limit errors and friendly error messages."""
    for attempt in range(max_retries):
        try:
            return agent.invoke(inputs, config=config) if config else agent.invoke(inputs)
        except Exception as e:
            err = str(e)
            # Rate limit — retry with backoff
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"\n{_YELLOW}Rate limited by the API. Waiting {wait}s before retry "
                      f"({attempt + 1}/{max_retries})...{_RESET}")
                time.sleep(wait)
                continue
            # Connection / timeout
            if "connect" in err.lower() or "timeout" in err.lower():
                print(f"\n{_RED}Connection error: Could not reach the API.{_RESET}")
                print("Check your internet connection. If using Ollama, make sure it's running: ollama serve")
                return None
            # Invalid API key
            if "403" in err or "API_KEY_INVALID" in err or "PERMISSION_DENIED" in err:
                print(f"\n{_RED}API key error: Your key may be invalid or expired.{_RESET}")
                print("Check your .env file or get a new key at https://aistudio.google.com/apikey")
                return None
            # Unknown error — don't retry
            print(f"\n{_RED}Error: {err}{_RESET}")
            return None

    print(f"\n{_RED}Still rate limited after {max_retries} retries. "
          f"Wait a minute and try again, or switch to Ollama.{_RESET}")
    return None


async def ainvoke_agent(agent, inputs, config=None, max_retries=3):
    """Async version of invoke_agent using agent.ainvoke().
    Use this in Chainlit or any other async context."""
    for attempt in range(max_retries):
        try:
            return await agent.ainvoke(inputs, config=config) if config else await agent.ainvoke(inputs)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"\n{_YELLOW}Rate limited by the API. Waiting {wait}s before retry "
                      f"({attempt + 1}/{max_retries})...{_RESET}")
                await asyncio.sleep(wait)
                continue
            if "connect" in err.lower() or "timeout" in err.lower():
                print(f"\n{_RED}Connection error: Could not reach the API.{_RESET}")
                print("Check your internet connection. If using Ollama, make sure it's running: ollama serve")
                return None
            if "403" in err or "API_KEY_INVALID" in err or "PERMISSION_DENIED" in err:
                print(f"\n{_RED}API key error: Your key may be invalid or expired.{_RESET}")
                print("Check your .env file or get a new key at https://aistudio.google.com/apikey")
                return None
            print(f"\n{_RED}Error: {err}{_RESET}")
            return None

    print(f"\n{_RED}Still rate limited after {max_retries} retries. "
          f"Wait a minute and try again, or switch to Ollama.{_RESET}")
    return None
