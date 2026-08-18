"""
Agentic AI Workshop - Dev/Test Crew
Flask web app version (deployable on Render).

The original notebook used input() in a loop, which only works in an
interactive terminal (like Colab). This version exposes the same
"Developer -> Tester" agent pipeline through a simple web form so it can
run as a hosted web service.
"""

import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from flask import Flask, request, render_template_string

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

llm = None
llm_init_error = None
if API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=API_KEY,
        )
    except Exception as e:
        llm_init_error = str(e)
else:
    llm_init_error = "GEMINI_API_KEY environment variable is not set."


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
rt_app = rt_workflow.compile()


# ==========================================
# 6. FLASK WEB APP
# ==========================================
app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Agentic Dev/Test Crew</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 16px; }
    textarea, input[type=text] { width: 100%; padding: 8px; font-size: 14px; }
    button { padding: 10px 20px; margin-top: 10px; cursor: pointer; }
    pre { background: #f4f4f4; padding: 12px; white-space: pre-wrap; word-wrap: break-word; border-radius: 6px; }
    .error { color: #b00020; }
    h2 { margin-top: 30px; }
  </style>
</head>
<body>
  <h1>Agentic AI Dev/Test Crew</h1>
  <p>Enter a coding task. The Developer agent writes code, the Tester agent generates test cases and runs it.</p>

  {% if error %}
    <p class="error">{{ error }}</p>
  {% endif %}

  <form method="POST">
    <label for="task">Coding task:</label>
    <textarea id="task" name="task" rows="3" placeholder="e.g. Write a function that checks if a number is prime">{{ task or '' }}</textarea>
    <button type="submit">Run Pipeline</button>
  </form>

  {% if code %}
    <h2>Generated Code</h2>
    <pre>{{ code }}</pre>
  {% endif %}

  {% if report %}
    <h2>Report</h2>
    <pre>{{ report }}</pre>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    code = None
    report = None
    error = None
    task = None

    if llm is None:
        error = f"LLM not configured: {llm_init_error}"

    if request.method == "POST" and llm is not None:
        task = request.form.get("task", "").strip()
        if not task:
            error = "Please enter a coding task."
        else:
            try:
                result = rt_app.invoke(
                    {"messages": [HumanMessage(content=task)]},
                    config={"recursion_limit": 50},
                )
                code = result.get("code")
                report = result.get("report")
            except Exception as e:
                error = f"Pipeline error: {e}"

    return render_template_string(
        PAGE_TEMPLATE, task=task, code=code, report=report, error=error
    )


@app.route("/health")
def health():
    return {"status": "ok", "llm_configured": llm is not None}


if __name__ == "__main__":
    # Render sets the PORT environment variable for you.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
