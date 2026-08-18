

import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. LLM INITIALIZATION
# ==========================================
# On Render, set GEMINI_API_KEY as an Environment Variable in the
# service settings (Dashboard -> your service -> Environment).
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it in Render's Environment tab before deploying."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=API_KEY
)


# ==========================================
# 2. STATE DEFINITION
# ==========================================
class CrewState(TypedDict):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]


# ==========================================
# 3. TOOLS
# ==========================================
@tool
def run_python_code(code: str) -> str:
    """Execute python code and return the standard output or error trace."""
    if not isinstance(code, str):
        code = str(code)
    clean_code = code.replace("```python", "").replace("```", "").strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""
    prompt = (
        f"You are a Senior QA Engineer. Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task: '{task_description}'.\n"
        f"Include standard cases and edge cases. Return them as a numbered list."
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def _extract_text(content):
    """Safely pull plain text out of Gemini's response content format."""
    if isinstance(content, list):
        first = content[0]
        return first.get("text", "") if isinstance(first, dict) else str(first)
    return str(content)


# ==========================================
# 4. GRAPH NODES (single-pass, no input())
# ==========================================
def real_time_developer(state: CrewState):
    task = state["messages"][-1].content
    dev_prompt = (
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )
    response = llm.invoke(dev_prompt)
    code_str = _extract_text(response.content)
    return {"code": code_str}


def real_time_tester(state: CrewState):
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    cases_str = _extract_text(test_cases)

    execution_result = run_python_code.invoke({"code": state["code"]})

    report = (
        f"### EXECUTION OUTPUT:\n{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED:\n{cases_str}"
    )
    return {"report": report}


# ==========================================
# 5. GRAPH CONSTRUCTION
# ==========================================
rt_workflow = StateGraph(CrewState)
rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)
rt_workflow.add_edge(START, "developer")
rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", END)

# Compiled LangGraph apps implement the LangChain Runnable interface,
# so LangServe can serve them directly as an API endpoint.
rt_app = rt_workflow.compile()


# ==========================================
# 6. SMALL WRAPPER RUNNABLE
# ==========================================
# LangServe input/output schemas are cleanest around simple types, so
# wrap the graph to accept a plain string task and return code + report.
from langchain_core.runnables import RunnableLambda


def _run_pipeline(task: str) -> dict:
    result = rt_app.invoke(
        {"messages": [HumanMessage(content=task)]},
        config={"recursion_limit": 50},
    )
    return {"code": result.get("code"), "report": result.get("report")}


pipeline_runnable = RunnableLambda(_run_pipeline)


# ==========================================
# 7. FASTAPI APP + LANGSERVE ROUTES
# ==========================================
app = FastAPI(
    title="Agentic Dev/Test Crew",
    version="1.0",
    description="LangGraph Developer/Tester pipeline served via LangServe.",
)


@app.get("/")
async def root():
    # LangServe auto-generates a playground UI at /pipeline/playground
    return RedirectResponse("/pipeline/playground")


@app.get("/health")
async def health():
    return {"status": "ok"}


# Exposes:
#   POST /pipeline/invoke   (single call)
#   POST /pipeline/batch
#   POST /pipeline/stream
#   GET  /pipeline/playground  (interactive test UI)
add_routes(app, pipeline_runnable, path="/pipeline")


if __name__ == "__main__":
    import uvicorn

    # Render sets the PORT environment variable for you.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
