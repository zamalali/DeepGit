import gradio as gr
import os
import json
import time
import threading
import logging
from agent import graph  # Your DeepGit langgraph workflow

# ---------------------------
# Global Logging Buffer Setup
# ---------------------------
LOG_BUFFER = []
LOG_BUFFER_LOCK = threading.Lock()

class BufferLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        with LOG_BUFFER_LOCK:
            LOG_BUFFER.append(log_entry)

# Attach the custom logging handler if not already attached.
root_logger = logging.getLogger()
if not any(isinstance(h, BufferLogHandler) for h in root_logger.handlers):
    handler = BufferLogHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

# ---------------------------
# Helper to Filter Log Messages
# ---------------------------
def filter_logs(logs):
    """
    Processes a list of log messages so that any log containing
    "HTTP Request:" is replaced with a generic message, and adjacent
    HTTP logs are deduplicated.
    """
    filtered = []
    last_was_fetching = False
    for log in logs:
        if "HTTP Request:" in log:
            if not last_was_fetching:
                filtered.append("Fetching repositories...")
                last_was_fetching = True
        else:
            filtered.append(log)
            last_was_fetching = False
    return filtered

# ---------------------------
# Title, Favicon & Description
# ---------------------------
#custom_theme = gr.Theme.load("gstaff/sketch")

favicon_html = """
<head>
<link rel="icon" type="image/x-icon" href="file/assets/deepgit.ico">
<title>DeepGit Research Agent</title>
</head>
"""

title = """
<div style="text-align: center; margin-top: 20px;">
  <h1 style="font-size: 36px; display: inline-flex; align-items: center; gap: 16px;">
    <img src="https://img.icons8.com/?size=100&id=118557&format=png&color=000000" width="64" />
    <span>DeepGit</span>
  </h1>
  <p style="font-size: 18px; color: #555; margin-top: 10px;">
    ⚙️ Built for open-source, by an open-sourcer — DeepGit finds gold in the GitHub haystack.
  </p>
</div>
"""

description = """<p align="center">
DeepGit is an agentic workflow built to perform advanced semantic research across GitHub repositories.<br/>
Enter your research topic below and let DeepGit intelligently analyze, rank, and explain the most relevant repositories for your query.<br/>
This may take a few minutes as DeepGit orchestrates multiple tools including Query Expansion, Semantic Retrieval, Cross-Encoder Ranking, Codebase Mapping, and Community Insight modules.
</p>"""

consent_text = """
<div style="padding: 10px; text-align: center;">
  <p>
    By using DeepGit, you consent to the collection and temporary processing of your query for semantic search and ranking purposes.<br/>
    No data is stored permanently, and your input is only used to power the DeepGit agent workflow.
  </p>
  <p>
    ⭐ Star us on GitHub if you find this tool useful!<br/>
    <a href="https://github.com/zamalali/DeepGit" target="_blank">GitHub</a>
  </p>
</div>
"""

footer = """
<div style="text-align: center; margin-top: 40px; font-size: 13px; color: #888;">
    Made with <span style="color: crimson;">❤️</span> by <b>Zamal</b>
</div>
"""

# ---------------------------
# HTML Table Renderer
# ---------------------------
def format_percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except:
        return value

def parse_result_to_html(raw_result: str) -> str:
    entries = raw_result.strip().split("Final Rank:")
    html = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 14px;
        }
        th, td {
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: left;
            vertical-align: top;
        }
        th {
            background-color: #f4f4f4;
        }
        tr:hover { background-color: #f9f9f9; }
    </style>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Title</th>
                <th>Link</th>
                <th>Semantic Similarity</th>
                <th>Cross-Encoder</th>
                <th>Final Score</th>
            </tr>
        </thead>
        <tbody>
    """
    for entry in entries[1:]:
        lines = entry.strip().split("\n")
        data = {}
        data["Final Rank"] = lines[0].strip()
        for line in lines[1:]:
            if ": " in line:
                key, val = line.split(": ", 1)
                data[key.strip()] = val.strip()
        html += f"""
            <tr>
                <td>{data.get('Final Rank', '')}</td>
                <td>{data.get('Title', '')}</td>
                <td><a href="{data.get('Link', '#')}" target="_blank">GitHub</a></td>
                <td>{format_percent(data.get('Semantic Similarity', ''))}</td>
                <td>{float(data.get('Cross-Encoder Score', 0)):.2f}</td>
                <td>{format_percent(data.get('Final Score', ''))}</td>
            </tr>
        """
    html += "</tbody></table>"
    return html

# ---------------------------
# Background Workflow Runner
# ---------------------------
def run_workflow(topic, result_container):
    """Runs the DeepGit workflow and stores the raw result."""
    initial_state = {"user_query": topic}
    result = graph.invoke(initial_state)
    result_container["raw_result"] = result.get("final_results", "No results returned.")

def stream_workflow(topic):
    # Clear the global log buffer
    with LOG_BUFFER_LOCK:
        LOG_BUFFER.clear()
    result_container = {}
    # Run the workflow in a background thread
    workflow_thread = threading.Thread(target=run_workflow, args=(topic, result_container))
    workflow_thread.start()
    
    last_index = 0
    # While the background thread is alive or new log messages are available, stream updates.
    while workflow_thread.is_alive() or (last_index < len(LOG_BUFFER)):
        with LOG_BUFFER_LOCK:
            new_logs = LOG_BUFFER[last_index:]
            last_index = len(LOG_BUFFER)
        if new_logs:
            # Filter the logs to replace HTTP request messages.
            filtered_logs = filter_logs(new_logs)
            status_msg = filtered_logs[-1]
            detail_msg = "<br/>".join(filtered_logs)
            yield status_msg, detail_msg
        time.sleep(0.5)
    
    workflow_thread.join()
    with LOG_BUFFER_LOCK:
        final_logs = LOG_BUFFER[:]
    filtered_final = filter_logs(final_logs)
    final_status = filtered_final[-1] if filtered_final else "Workflow completed."
    raw_result = result_container.get("raw_result", "No results returned.")
    html_result = parse_result_to_html(raw_result)
    yield "", html_result

# ---------------------------
# App UI Setup
# ---------------------------
#  To change the theme set: theme="gstaff/sketch",
with gr.Blocks(
    theme="gstaff/sketch",
    css="""
        #main_container { margin: auto; max-width: 900px; }
        footer, footer * {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            overflow: hidden !important;
        }
    """
) as demo:

    gr.HTML(favicon_html)
    gr.HTML(title)
    gr.HTML(description)

    with gr.Column(elem_id="user_consent_container") as consent_block:
        gr.HTML(consent_text)
        agree_button = gr.Button("I Agree", variant="primary")

    with gr.Column(elem_id="main_container", visible=False) as main_block:
        research_input = gr.Textbox(
            label="Research Topic",
            placeholder="Enter your research topic here, e.g., 'Instruction-based fine-tuning for LLaMA 2 using chain-of-thought prompting in Python.' ",
            lines=3
        )
        run_button = gr.Button("Run DeepGit", variant="primary")
        # Display the latest log line as status, and full log stream as details.
        status_display = gr.Markdown("")   
        detail_display = gr.HTML("")
        output_html = gr.HTML()
        state = gr.State([])

    def enable_main():
        return gr.update(visible=False), gr.update(visible=True)

    agree_button.click(fn=enable_main, inputs=[], outputs=[consent_block, main_block], queue=False)

    # Generator-based runner for dynamic log streaming.
    def stepwise_runner(topic):
        for status, details in stream_workflow(topic):
            yield status, details

    run_button.click(
        fn=stepwise_runner,
        inputs=[research_input],
        outputs=[status_display, detail_display],
        api_name="deepgit",
        show_progress=True
    )

    research_input.submit(
        fn=stepwise_runner,
        inputs=[research_input],
        outputs=[status_display, detail_display],
        api_name="deepgit_submit",
        show_progress=True
    )

    gr.HTML(footer)
demo.queue(max_size=10).launch()

