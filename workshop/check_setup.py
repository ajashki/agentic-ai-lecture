#!/usr/bin/env python3
"""
Pre-workshop environment checker for the Agentic AI workshop.
Run this to verify your setup is ready.

Works on macOS, Linux, and Windows.
Can be run standalone (before cloning the repo) or from within the workshop directory.
"""

import os
import sys
import platform
import shutil
import subprocess

# Enable ANSI colors on Windows 10+
if sys.platform == "win32":
    os.system("")  # enables ANSI escape sequences on Windows terminal

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"
CHECK = f"{GREEN}\u2713{RESET}"
CROSS = f"{RED}\u2717{RESET}"
WARN = f"{YELLOW}!{RESET}"

issues = []
warnings = []


def check(label, passed, fix=""):
    if passed:
        print(f"  {CHECK} {label}")
    else:
        print(f"  {CROSS} {label}")
        if fix:
            print(f"      Fix: {fix}")
        issues.append(label)


def warn(label, msg=""):
    print(f"  {WARN} {label}")
    if msg:
        print(f"      {msg}")
    warnings.append(label)


print(f"\n{BOLD}=== Agentic AI Workshop — Setup Check ==={RESET}")
print(f"  Platform: {platform.system()} {platform.release()}\n")

# 1. Python version
v = sys.version_info
check(
    f"Python {v.major}.{v.minor}.{v.micro}",
    v >= (3, 10),
    "Install Python 3.10+ from https://www.python.org/downloads/",
)

# 2. pip
pip_available = shutil.which("pip3") or shutil.which("pip")
if not pip_available:
    if sys.platform == "linux":
        fix = "Run: sudo apt install python3-pip (Ubuntu/Debian) or sudo dnf install python3-pip (Fedora)"
    else:
        fix = "Try: python -m ensurepip --upgrade. If that fails, reinstall Python from https://www.python.org/downloads/"
else:
    fix = ""
check("pip available", pip_available is not None, fix)

# 3. git
git_available = shutil.which("git")
if git_available:
    check("git available", True)
else:
    if sys.platform == "win32":
        fix = "Install from https://git-scm.com/download/win"
    elif sys.platform == "darwin":
        fix = "Run: xcode-select --install (or install from https://git-scm.com)"
    else:
        fix = "Run: sudo apt install git (Ubuntu/Debian) or sudo dnf install git (Fedora)"
    check("git available", False, fix)

# 4. Core packages
print()
missing_packages = []
for pkg, import_name in [
    ("langgraph", "langgraph"),
    ("langchain-google-genai", "langchain_google_genai"),
    ("langchain-core", "langchain_core"),
    ("chainlit", "chainlit"),
    ("python-dotenv", "dotenv"),
]:
    try:
        __import__(import_name)
        check(f"{pkg} installed", True)
    except ImportError:
        check(f"{pkg} installed", False)
        missing_packages.append(pkg)

if missing_packages:
    print(f"\n      To install all missing packages:")
    print(f"      pip install {' '.join(missing_packages)}")
    print(f"      Or: pip install -r requirements.txt (if you have the repo)")

# 6. API key
print()
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    check("GOOGLE_API_KEY found", True)
else:
    # Check if .env file exists
    env_exists = os.path.exists(".env")
    if env_exists:
        check(
            "GOOGLE_API_KEY found",
            False,
            "Key not set in .env file. Paste your key after GOOGLE_API_KEY= (no spaces around =)",
        )
    else:
        check(
            "GOOGLE_API_KEY found",
            False,
            "Create a .env file with: GOOGLE_API_KEY=your-key-here\n"
            "      Get a free key at https://aistudio.google.com/apikey",
        )

# 7. Test Gemini connection
if api_key:
    print(f"\n  Testing Gemini connection...")
    try:
        from config import get_llm

        llm = get_llm()
        response = llm.invoke("Say 'hello' and nothing else.")
        check("Gemini API connection works", True)
    except ImportError:
        # config.py not available (running standalone before cloning repo)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
                google_api_key=api_key,
            )
            response = llm.invoke("Say 'hello' and nothing else.")
            check("Gemini API connection works", True)
        except Exception as e:
            err = str(e)
            if "403" in err or "API_KEY_INVALID" in err:
                fix = "Your API key is invalid. Generate a new one at https://aistudio.google.com/apikey"
            elif "429" in err:
                fix = "Rate limited. Wait 60 seconds and try again."
            elif "SSL" in err or "certificate" in err.lower():
                fix = "SSL error — your network may block API calls. Try a different network or use Ollama fallback."
            elif "connect" in err.lower() or "timeout" in err.lower():
                fix = "Cannot reach Gemini API. Check your internet connection, or your network may be blocking it."
            else:
                fix = f"Error: {err}"
            check("Gemini API connection works", False, fix)
    except Exception as e:
        err = str(e)
        if "403" in err or "API_KEY_INVALID" in err:
            fix = "Your API key is invalid. Generate a new one at https://aistudio.google.com/apikey"
        elif "429" in err:
            fix = "Rate limited. Wait 60 seconds and try again."
        elif "SSL" in err or "certificate" in err.lower():
            fix = "SSL error — your network may block API calls. Try a different network or use Ollama fallback."
        elif "connect" in err.lower() or "timeout" in err.lower():
            fix = "Cannot reach Gemini API. Check your internet connection, or your network may be blocking it."
        else:
            fix = f"Error: {err}"
        check("Gemini API connection works", False, fix)
else:
    print(f"\n  {YELLOW}Skipping Gemini test (no API key){RESET}")

# 8. Test Ollama (optional)
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
print(f"\n  Testing Ollama (optional fallback)...")
ollama_available = shutil.which("ollama")
if ollama_available:
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=ollama_model)
        response = llm.invoke("Say 'hello' and nothing else.")
        check(f"Ollama + {ollama_model} works", True)
    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "refused" in err.lower():
            warn(
                "Ollama installed but not running",
                "Start it with: ollama serve (in a separate terminal)",
            )
        elif "not found" in err.lower() or "404" in err.lower():
            warn(
                f"Ollama running but {ollama_model} model not pulled",
                f"Run: ollama pull {ollama_model}",
            )
        else:
            warn(f"Ollama error: {err}")
else:
    print(f"  {YELLOW}- Ollama not installed (optional — only needed if Gemini doesn't work){RESET}")
    if sys.platform == "win32":
        print(f"    Install from https://ollama.com/download/windows")
    elif sys.platform == "darwin":
        print(f"    Install from https://ollama.com or: brew install ollama")
    else:
        print(f"    Install from https://ollama.com")
    print(f"    Then run: ollama pull {ollama_model}")

# 9. Port 8000 check (for Chainlit in Phase 3)
print()
try:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", 8000))
    sock.close()
    if result == 0:
        warn(
            "Port 8000 is already in use",
            "Chainlit uses port 8000. You may need to stop whatever is running there, "
            "or use: chainlit run phase3_chat.py --port 8001",
        )
    else:
        check("Port 8000 available (for Chainlit)", True)
except Exception:
    check("Port 8000 available (for Chainlit)", True)

# Summary
print()
if not issues and not warnings:
    print(f"{GREEN}{'=' * 48}")
    print(f"  All checks passed! You're ready for the workshop.")
    print(f"{'=' * 48}{RESET}\n")
elif not issues:
    print(f"{YELLOW}{'=' * 48}")
    print(f"  All required checks passed ({len(warnings)} warning(s)).")
    print(f"  You're good to go — warnings are optional.")
    print(f"{'=' * 48}{RESET}\n")
else:
    print(f"{RED}{'=' * 48}")
    print(f"  {len(issues)} issue(s) need fixing. See above.")
    if warnings:
        print(f"  ({len(warnings)} warning(s) — optional)")
    print(f"{'=' * 48}{RESET}")
    print(f"\n  If you're stuck, arrive 10 minutes early and we'll help!\n")
    sys.exit(1)
