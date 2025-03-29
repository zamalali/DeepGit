import gradio as gr
import os
import json
import time
from agent import graph  # Your DeepGit langgraph workflow
import spaces  # Enables @spaces.GPU decorator

# Load the custom theme
custom_theme = gr.Theme.load("themes/theme_schema@0.0.1.json")

# ---------------------------
# Title, Favicon & Description
# ---------------------------
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


# ---------------------------
# Consent Text & Footer
# ---------------------------
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
# Streaming Workflow with Logging Sync
# ---------------------------

@spaces.GPU
def run_deepgit_on_gpu(research_topic):
    initial_state = {"user_query": research_topic}
    result = graph.invoke(initial_state)  # heavy task
    raw_result = result.get("final_results", "No results returned.")
    return parse_result_to_html(raw_result)

def run_deepgit_steps(research_topic):
    steps = [
        ("🧠 Expanding your query...", 1.0),
        ("🔍 Retrieving from GitHub...", 1.5),
        ("📚 Removing duplicates & preparing data...", 1.0),
        ("🧬 Embedding with SentenceTransformer...", 1.5),
        ("📈 Running Dense Retrieval...", 2.5),
        ("🧠 Re-ranking with Cross Encoder...", 3.0),
        ("✨ Finalizing results...", 1.0),
    ]

    for msg, delay in steps:
        yield msg, ""
        time.sleep(delay)

    yield "🛠️ DeepGit Agent is working...", ""
    html_result = run_deepgit_on_gpu(research_topic)
    yield "", html_result


# ---------------------------
# App UI
# ---------------------------
with gr.Blocks(
    theme=custom_theme,
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
            placeholder="Enter your research topic here...",
            lines=3
        )
        run_button = gr.Button("Run DeepGit", variant="primary")
        status_display = gr.Markdown("")  # Dynamic progress
        output_html = gr.HTML()
        state = gr.State([])

    def enable_main():
        return gr.update(visible=False), gr.update(visible=True)

    agree_button.click(fn=enable_main, inputs=[], outputs=[consent_block, main_block], queue=False)

    def stepwise_runner(topic):
        for step_text, html_out in run_deepgit_steps(topic):
            yield step_text, html_out

    run_button.click(
        fn=stepwise_runner,
        inputs=[research_input],
        outputs=[status_display, output_html],
        api_name="deepgit",
        show_progress=True
    )

    research_input.submit(
        fn=stepwise_runner,
        inputs=[research_input],
        outputs=[status_display, output_html],
        api_name="deepgit_submit",
        show_progress=True
    )

    gr.HTML(footer)
    demo.queue(max_size=10).launch(share=True)

if __name__ == "__main__":
    demo.launch(share=True)
