import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from langserve import add_routes

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. GEMINI INITIALIZATION
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it in Render Environment Variables."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=API_KEY,
    temperature=0
)


# ============================================================
# 2. STATE
# ============================================================

class CrewState(TypedDict):

    messages: List[BaseMessage]

    code: Optional[str]

    test_cases: Optional[str]

    execution_result: Optional[str]

    final_report: Optional[str]


# ============================================================
# 3. PYTHON EXECUTION TOOL
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute generated Python code and return its output.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout

    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:

        local_scope = {}

        exec(
            clean_code,
            {},
            local_scope
        )

        result = new_stdout.getvalue()

    except Exception:

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    if result.strip():

        return result.strip()

    return "Execution completed successfully with no terminal output."


# ============================================================
# 4. TEST CASE GENERATOR
# ============================================================

@tool
def generate_test_cases(
    task_description: str
) -> str:
    """
    Generate professional test scenarios for a coding task.
    """

    prompt = f"""
You are a Senior QA Engineer.

Create exactly 4 professional test scenarios for this
coding task:

{task_description}

For each test case provide:

Test Case Number
Test Objective
Input
Expected Result
Test Type

Include:

1. Base case
2. Normal case
3. Edge case
4. Invalid or boundary case

Do not write Python code.

Return only the test cases.
"""

    response = llm.invoke(prompt)

    if hasattr(response, "content"):

        return response.content

    return str(response)


# ============================================================
# 5. TEXT HELPER
# ============================================================

def extract_text(content):

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get(
                    "text",
                    ""
                )

                if text:
                    parts.append(text)

            else:

                parts.append(
                    str(item)
                )

        return "\n".join(parts).strip()

    return str(content).strip()


# ============================================================
# 6. DEVELOPER NODE
# ============================================================

def real_time_developer(
    state: CrewState
):

    task = state[
        "messages"
    ][-1].content


    developer_prompt = f"""
You are the Developer Agent in a professional
AI software development pipeline.

User's coding task:

{task}

Generate a clean, executable Python solution.

Requirements:

- Write complete Python code.
- Use clear variable names.
- Keep the implementation simple and correct.
- Include a small executable test/demo at the bottom.
- The program must print useful output.
- Do not use markdown.
- Do not include explanations.
- Return ONLY Python code.
"""


    response = llm.invoke(
        developer_prompt
    )


    code_str = extract_text(
        response.content
    )


    return {

        "code": code_str

    }


# ============================================================
# 7. TESTER NODE
# ============================================================

def real_time_tester(
    state: CrewState
):

    task = state[
        "messages"
    ][-1].content


    test_cases = (
        generate_test_cases.invoke(
            task
        )
    )


    cases_str = extract_text(
        test_cases
    )


    return {

        "test_cases":
            cases_str

    }


# ============================================================
# 8. EXECUTOR NODE
# ============================================================

def real_time_executor(
    state: CrewState
):

    code = state.get(
        "code",
        ""
    )


    execution_result = (
        run_python_code.invoke(
            {
                "code": code
            }
        )
    )


    return {

        "execution_result":
            execution_result

    }


# ============================================================
# 9. FINAL REPORT NODE
# ============================================================

def generate_final_report(
    state: CrewState
):

    task = state[
        "messages"
    ][-1].content


    code = state.get(
        "code",
        ""
    )


    test_cases = state.get(
        "test_cases",
        ""
    )


    execution_result = state.get(
        "execution_result",
        ""
    )


    report_prompt = f"""
You are the final reporting agent in an AI software
development pipeline.

Create a concise professional report.

USER TASK:
{task}

DEVELOPER OUTPUT:
{code}

TESTER OUTPUT:
{test_cases}

EXECUTOR OUTPUT:
{execution_result}

Return the report using exactly these sections:

DEVELOPMENT STATUS
TESTING STATUS
EXECUTION STATUS
SUMMARY

Do not invent results.

If execution completed successfully, say so.

If there was an execution error, clearly mention it.
"""


    response = llm.invoke(
        report_prompt
    )


    report = extract_text(
        response.content
    )


    return {

        "final_report":
            report

    }


# ============================================================
# 10. LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(
    CrewState
)


workflow.add_node(
    "developer",
    real_time_developer
)


workflow.add_node(
    "tester",
    real_time_tester
)


workflow.add_node(
    "executor",
    real_time_executor
)


workflow.add_node(
    "reporter",
    generate_final_report
)


workflow.add_edge(
    START,
    "developer"
)


workflow.add_edge(
    "developer",
    "tester"
)


workflow.add_edge(
    "tester",
    "executor"
)


workflow.add_edge(
    "executor",
    "reporter"
)


workflow.add_edge(
    "reporter",
    END
)


rt_app = workflow.compile()


# ============================================================
# 11. PIPELINE RUNNER
# ============================================================

def _run_pipeline(
    task: str
):

    result = rt_app.invoke(

        {
            "messages": [
                HumanMessage(
                    content=task
                )
            ]
        },

        config={
            "recursion_limit": 50
        }
    )


    return {

        "task":
            task,

        "code":
            result.get(
                "code",
                ""
            ),

        "test_cases":
            result.get(
                "test_cases",
                ""
            ),

        "execution_result":
            result.get(
                "execution_result",
                ""
            ),

        "final_report":
            result.get(
                "final_report",
                ""
            )

    }


pipeline_runnable = RunnableLambda(
    _run_pipeline
)


# ============================================================
# 12. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Agentic Developer Tester Executor",

    version="2.0",

    description=(
        "Professional AI Developer, Tester and "
        "Executor pipeline."
    )
)


# ============================================================
# 13. CUSTOM PROFESSIONAL UI
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>DevCrew AI</title>


<style>

/* ==========================================================
   GLOBAL
========================================================== */

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    font-family:
        Inter,
        Segoe UI,
        Arial,
        sans-serif;

    color: #172033;

    min-height: 100vh;

    background:
        radial-gradient(
            circle at top left,
            rgba(99,102,241,0.16),
            transparent 32%
        ),

        radial-gradient(
            circle at bottom right,
            rgba(14,165,233,0.14),
            transparent 30%
        ),

        #f5f7fb;
}


/* ==========================================================
   HEADER
========================================================== */

.header {

    height: 72px;

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding:
        0 5%;

    background:
        rgba(255,255,255,0.86);

    backdrop-filter:
        blur(15px);

    border-bottom:
        1px solid #e6e9f0;

    position:
        sticky;

    top: 0;

    z-index: 100;
}


.brand {

    display: flex;

    align-items: center;

    gap: 13px;
}


.logo {

    width: 43px;

    height: 43px;

    border-radius: 13px;

    display: flex;

    align-items: center;

    justify-content: center;

    color: white;

    font-weight: 800;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );

    box-shadow:
        0 8px 20px
        rgba(79,70,229,0.25);
}


.brand h1 {

    margin: 0;

    font-size: 19px;

}


.brand span {

    color: #667085;

    font-size: 11px;

}


.status {

    display: flex;

    align-items: center;

    gap: 7px;

    padding:
        8px 13px;

    border-radius: 20px;

    background: #ecfdf3;

    color: #15803d;

    font-size: 11px;

    font-weight: 600;

}


.status-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #22c55e;

}


/* ==========================================================
   CONTAINER
========================================================== */

.container {

    max-width: 1200px;

    margin:
        0 auto;

    padding:
        45px 20px 60px;
}


/* ==========================================================
   HERO
========================================================== */

.hero {

    text-align: center;

    margin-bottom: 40px;

}


.badge {

    display: inline-block;

    padding:
        7px 13px;

    border-radius: 30px;

    background: #eef2ff;

    color: #4f46e5;

    font-size: 11px;

    font-weight: 700;

    margin-bottom: 15px;

}


.hero h2 {

    font-size:
        clamp(30px, 5vw, 48px);

    margin:
        0 0 12px;

    letter-spacing:
        -1.5px;

}


.hero p {

    max-width: 680px;

    margin: auto;

    color: #667085;

    line-height: 1.7;

    font-size: 14px;

}


/* ==========================================================
   PIPELINE
========================================================== */

.pipeline {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 12px;

    margin-bottom: 30px;

}


.stage {

    background: white;

    border:
        1px solid #e6e9f0;

    border-radius: 16px;

    padding: 17px;

    text-align: center;

    transition:
        0.25s ease;

}


.stage.active {

    border-color:
        #818cf8;

    box-shadow:
        0 10px 30px
        rgba(99,102,241,0.12);

    transform:
        translateY(-2px);

}


.stage-number {

    width: 35px;

    height: 35px;

    margin:
        0 auto 9px;

    border-radius: 11px;

    display: flex;

    align-items: center;

    justify-content: center;

    background: #eef2ff;

    color: #4f46e5;

    font-weight: 800;

}


.stage strong {

    display: block;

    font-size: 13px;

}


.stage small {

    display: block;

    color: #98a2b3;

    margin-top: 4px;

    font-size: 10px;

}


/* ==========================================================
   INPUT CARD
========================================================== */

.input-card {

    background: white;

    border:
        1px solid #e6e9f0;

    border-radius: 22px;

    padding: 25px;

    box-shadow:
        0 15px 40px
        rgba(16,24,40,0.06);

    margin-bottom: 25px;

}


.input-card label {

    display: block;

    font-size: 13px;

    font-weight: 700;

    margin-bottom: 10px;

}


.input-area {

    display: flex;

    gap: 10px;

}


textarea {

    flex: 1;

    min-height: 70px;

    resize: vertical;

    padding: 15px;

    border:
        1px solid #dfe3eb;

    border-radius: 14px;

    outline: none;

    font:
        inherit;

    font-size: 13px;

}


textarea:focus {

    border-color:
        #818cf8;

    box-shadow:
        0 0 0 4px
        rgba(99,102,241,0.08);

}


.run-button {

    min-width: 125px;

    border: none;

    border-radius: 14px;

    color: white;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );

    font-weight: 700;

    cursor: pointer;

    transition:
        0.2s ease;

}


.run-button:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 10px 25px
        rgba(79,70,229,0.25);

}


.run-button:disabled {

    opacity: 0.55;

    cursor: not-allowed;

}


.hint {

    margin-top: 8px;

    font-size: 10px;

    color: #98a2b3;

}


/* ==========================================================
   RESULT AREA
========================================================== */

.results {

    display: grid;

    gap: 18px;

}


/* ==========================================================
   RESULT CARD
========================================================== */

.result-card {

    background: white;

    border:
        1px solid #e6e9f0;

    border-radius: 20px;

    overflow: hidden;

    box-shadow:
        0 10px 30px
        rgba(16,24,40,0.05);

}


.result-header {

    display: flex;

    align-items: center;

    gap: 12px;

    padding:
        17px 20px;

    border-bottom:
        1px solid #edf0f4;

}


.result-icon {

    width: 37px;

    height: 37px;

    border-radius: 11px;

    display: flex;

    align-items: center;

    justify-content: center;

    background: #eef2ff;

    font-size: 17px;

}


.result-header strong {

    font-size: 14px;

}


.result-header span {

    display: block;

    color: #98a2b3;

    font-size: 10px;

    margin-top: 3px;

}


.result-content {

    padding: 20px;

}


.code {

    background:
        #111827;

    color:
        #e5e7eb;

    border-radius:
        13px;

    padding:
        18px;

    overflow-x:
        auto;

    font-family:
        Consolas,
        Monaco,
        monospace;

    font-size:
        12px;

    line-height:
        1.7;

    white-space:
        pre-wrap;

}


.test {

    white-space:
        pre-wrap;

    line-height:
        1.7;

    font-size:
        12px;

    color:
        #475467;

}


.execution {

    background:
        #0f172a;

    color:
        #d1fae5;

    border-radius:
        13px;

    padding:
        18px;

    font-family:
        Consolas,
        Monaco,
        monospace;

    font-size:
        12px;

    white-space:
        pre-wrap;

    line-height:
        1.7;

}


.report {

    white-space:
        pre-wrap;

    font-size:
        13px;

    line-height:
        1.75;

    color:
        #344054;

}


/* ==========================================================
   EMPTY STATE
========================================================== */

.empty {

    text-align:
        center;

    padding:
        50px 20px;

    color:
        #98a2b3;

}


.empty-icon {

    font-size:
        38px;

    margin-bottom:
        10px;

}


/* ==========================================================
   LOADING
========================================================== */

.loading {

    display:
        none;

    text-align:
        center;

    padding:
        35px;

}


.loader {

    width:
        35px;

    height:
        35px;

    margin:
        auto auto 12px;

    border:
        3px solid #e5e7eb;

    border-top-color:
        #4f46e5;

    border-radius:
        50%;

    animation:
        spin 0.8s linear infinite;

}


@keyframes spin {

    to {
        transform:
            rotate(360deg);
    }

}


.loading p {

    margin:
        0;

    font-size:
        12px;

    color:
        #667085;

}


/* ==========================================================
   RESPONSIVE
========================================================== */

@media(max-width: 800px) {

    .pipeline {

        grid-template-columns:
            repeat(2, 1fr);

    }


    .input-area {

        flex-direction:
            column;

    }


    .run-button {

        min-height:
            48px;

    }

}


@media(max-width: 480px) {

    .pipeline {

        grid-template-columns:
            1fr;

    }


    .header {

        padding:
            0 15px;

    }


    .status {

        display:
            none;

    }

}

</style>

</head>


<body>


<!-- ========================================================
     HEADER
======================================================== -->

<header class="header">

    <div class="brand">

        <div class="logo">
            DC
        </div>

        <div>

            <h1>
                DevCrew AI
            </h1>

            <span>
                Developer • Tester • Executor
            </span>

        </div>

    </div>


    <div class="status">

        <span class="status-dot"></span>

        AI Pipeline Online

    </div>

</header>


<!-- ========================================================
     MAIN
======================================================== -->

<main class="container">


    <section class="hero">

        <div class="badge">
            ✦ AGENTIC SOFTWARE ENGINEERING
        </div>

        <h2>
            AI Developer & QA Crew
        </h2>

        <p>
            Submit a coding task and let the AI development
            pipeline generate the solution, design test
            scenarios, execute the generated program and
            produce a professional engineering report.
        </p>

    </section>


    <!-- PIPELINE -->

    <section class="pipeline">


        <div
            class="stage"
            id="stageDeveloper"
        >

            <div class="stage-number">
                01
            </div>

            <strong>
                Developer
            </strong>

            <small>
                Generate solution
            </small>

        </div>


        <div
            class="stage"
            id="stageTester"
        >

            <div class="stage-number">
                02
            </div>

            <strong>
                Tester
            </strong>

            <small>
                Design test cases
            </small>

        </div>


        <div
            class="stage"
            id="stageExecutor"
        >

            <div class="stage-number">
                03
            </div>

            <strong>
                Executor
            </strong>

            <small>
                Run generated code
            </small>

        </div>


        <div
            class="stage"
            id="stageReporter"
        >

            <div class="stage-number">
                04
            </div>

            <strong>
                Reporter
            </strong>

            <small>
                Generate final report
            </small>

        </div>

    </section>


    <!-- INPUT -->

    <section class="input-card">

        <label>
            Coding Task
        </label>


        <div class="input-area">

            <textarea
                id="task"
                placeholder="Example: Write a Python program to calculate the factorial of a number."
            ></textarea>


            <button
                class="run-button"
                id="runButton"
                onclick="runPipeline()"
            >
                ▶ Run Pipeline
            </button>

        </div>


        <div class="hint">

            Press Ctrl + Enter to run the pipeline.

        </div>

    </section>


    <!-- LOADING -->

    <div
        class="loading"
        id="loading"
    >

        <div class="loader"></div>

        <p>
            AI crew is processing your task...
        </p>

    </div>


    <!-- RESULTS -->

    <section
        class="results"
        id="results"
    >

        <div class="empty">

            <div class="empty-icon">
                🤖
            </div>

            Submit a coding task to start the
            Developer → Tester → Executor pipeline.

        </div>

    </section>


</main>


<script>


// ==========================================================
// ELEMENTS
// ==========================================================

const taskInput =
    document.getElementById("task");

const runButton =
    document.getElementById("runButton");

const loading =
    document.getElementById("loading");

const results =
    document.getElementById("results");


// ==========================================================
// STAGES
// ==========================================================

const stages = [

    document.getElementById(
        "stageDeveloper"
    ),

    document.getElementById(
        "stageTester"
    ),

    document.getElementById(
        "stageExecutor"
    ),

    document.getElementById(
        "stageReporter"
    )

];


function activateStage(index) {

    stages.forEach(
        (stage, i) => {

            if (i <= index) {

                stage.classList.add(
                    "active"
                );

            } else {

                stage.classList.remove(
                    "active"
                );

            }

        }
    );

}


// ==========================================================
// RUN PIPELINE
// ==========================================================

async function runPipeline() {


    const task =
        taskInput.value.trim();


    if (!task) {

        alert(
            "Please enter a coding task."
        );

        taskInput.focus();

        return;

    }


    // Reset UI

    results.innerHTML = "";

    loading.style.display =
        "block";

    runButton.disabled =
        true;


    stages.forEach(
        stage =>
            stage.classList.remove(
                "active"
            )
    );


    try {


        // --------------------------------------------------
        // Developer
        // --------------------------------------------------

        activateStage(0);


        await delay(300);


        // --------------------------------------------------
        // Tester
        // --------------------------------------------------

        activateStage(1);


        await delay(300);


        // --------------------------------------------------
        // Executor
        // --------------------------------------------------

        activateStage(2);


        await delay(300);


        // --------------------------------------------------
        // API CALL
        // --------------------------------------------------

        const response =
            await fetch(
                "/run",
                {

                    method:
                        "POST",

                    headers:
                        {
                            "Content-Type":
                                "application/json"
                        },

                    body:
                        JSON.stringify({
                            task: task
                        })

                }
            );


        if (!response.ok) {

            throw new Error(
                "Server returned "
                + response.status
            );

        }


        const data =
            await response.json();


        // --------------------------------------------------
        // Reporter
        // --------------------------------------------------

        activateStage(3);


        await delay(250);


        // --------------------------------------------------
        // DISPLAY
        // --------------------------------------------------

        displayResults(
            data
        );


    } catch (error) {


        console.error(
            error
        );


        results.innerHTML = `

            <div class="result-card">

                <div class="result-header">

                    <div class="result-icon">
                        ⚠️
                    </div>

                    <div>

                        <strong>
                            Pipeline Error
                        </strong>

                        <span>
                            Something went wrong
                        </span>

                    </div>

                </div>

                <div class="result-content">

                    <div class="test">
                        ${escapeHtml(
                            error.message
                        )}
                    </div>

                </div>

            </div>

        `;


    } finally {

        loading.style.display =
            "none";

        runButton.disabled =
            false;

    }

}


// ==========================================================
// DISPLAY RESULTS
// ==========================================================

function displayResults(
    data
) {


    results.innerHTML = `

        <!-- DEVELOPER -->

        <div class="result-card">

            <div class="result-header">

                <div class="result-icon">
                    👨‍💻
                </div>

                <div>

                    <strong>
                        Developer Output
                    </strong>

                    <span>
                        Generated Python solution
                    </span>

                </div>

            </div>


            <div class="result-content">

                <div class="code">${escapeHtml(
                    data.code || "No code generated."
                )}</div>

            </div>

        </div>


        <!-- TESTER -->

        <div class="result-card">

            <div class="result-header">

                <div class="result-icon">
                    🧪
                </div>

                <div>

                    <strong>
                        Tester Output
                    </strong>

                    <span>
                        Generated QA scenarios
                    </span>

                </div>

            </div>


            <div class="result-content">

                <div class="test">${escapeHtml(
                    data.test_cases ||
                    "No test cases generated."
                )}</div>

            </div>

        </div>


        <!-- EXECUTOR -->

        <div class="result-card">

            <div class="result-header">

                <div class="result-icon">
                    ⚙️
                </div>

                <div>

                    <strong>
                        Executor Output
                    </strong>

                    <span>
                        Generated code execution result
                    </span>

                </div>

            </div>


            <div class="result-content">

                <div class="execution">${escapeHtml(
                    data.execution_result ||
                    "No execution result."
                )}</div>

            </div>

        </div>


        <!-- REPORT -->

        <div class="result-card">

            <div class="result-header">

                <div class="result-icon">
                    📊
                </div>

                <div>

                    <strong>
                        Final Engineering Report
                    </strong>

                    <span>
                        AI-generated pipeline summary
                    </span>

                </div>

            </div>


            <div class="result-content">

                <div class="report">${escapeHtml(
                    data.final_report ||
                    "No final report generated."
                )}</div>

            </div>

        </div>

    `;

}


// ==========================================================
// ESCAPE HTML
// ==========================================================

function escapeHtml(
    text
) {

    const div =
        document.createElement(
            "div"
        );

    div.textContent =
        String(text);

    return div.innerHTML;

}


// ==========================================================
// DELAY
// ==========================================================

function delay(
    milliseconds
) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                milliseconds
            )
    );

}


// ==========================================================
// CTRL + ENTER
// ==========================================================

taskInput.addEventListener(
    "keydown",
    function(event) {

        if (
            event.ctrlKey
            &&
            event.key === "Enter"
        ) {

            runPipeline();

        }

    }
);


</script>


</body>

</html>
"""


# ============================================================
# 14. ROOT UI
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def root():

    return HTML


# ============================================================
# 15. CUSTOM PIPELINE API
# ============================================================

@app.post("/run")
async def run_pipeline_api(
    payload: dict
):

    task = payload.get(
        "task",
        ""
    ).strip()


    if not task:

        return {
            "error":
                "Please provide a coding task."
        }


    return _run_pipeline(
        task
    )


# ============================================================
# 16. HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service":
            "DevCrew AI"
    }


# ============================================================
# 17. LANGSERVE API
# ============================================================

add_routes(

    app,

    pipeline_runnable,

    path="/pipeline"

)


# ============================================================
# 18. SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )


    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port

    )