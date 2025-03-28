import gradio as gr
from gradio import ChatMessage
from typing import List

# Import your agent's graph and state input from agent.py
from agent import graph, AgentStateInput

def run_deepgit_interface(prompt: str):
    """
    Runs the full DeepGit agent with the user query and returns the final results,
    logs, and a node status HTML snippet.
    """
    # Create an initial state from the prompt.
    initial_state = AgentStateInput(user_query=prompt)
    # Execute the full agent graph.
    # Try calling the graph directly (if the compiled graph is callable)
    result = graph(initial_state)
    # Alternatively, if that doesn't work, try:
    # result = graph.execute(initial_state)
    
    # Process the result as before...
    if isinstance(result, dict) and "final_results" in result:
        final_output = result["final_results"]
    else:
        final_output = "\n".join(
            [f"Title: {repo.get('title', 'N/A')} | Stars: {repo.get('stars', 0)} | Final Score: {repo.get('final_score', 0):.4f}"
             for repo in result.final_ranked]
        )
    
    logs = "Agent finished processing."
    node_status = "<div style='padding:5px;'><b>Final Node Status:</b> Completed all nodes.</div>"
    chat_history = [{"role": "assistant", "content": final_output}]
    return chat_history, logs, node_status


# --------------------------------------------------
# Gradio App Layout
# --------------------------------------------------
with gr.Blocks(css="""
/* Custom CSS for enhanced style */
#node-status {
    background-color: #e8f5e9;
    border: 2px solid #4CAF50;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
    font-weight: bold;
}
#log-output {
    background-color: #1e1e1e;
    color: #00ff00;
    font-family: monospace;
    padding: 10px;
    border-radius: 8px;
    height: 300px;
    overflow-y: auto;
}
""") as demo:
    gr.Markdown("# DeepGit: AI GitHub Researcher")
    gr.Markdown("### Your personal agent for finding the best GitHub repositories")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Chatbot interface that shows final results from the agent
            chatbot = gr.Chatbot(
                label="🧠 DeepGit Agent",
                avatar_images=(None, "https://emojicdn.elk.sh/1f4bb.png")
            )
            msg = gr.Textbox(label="Enter your query", placeholder="e.g. Find Python libraries for semantic search in PDFs")
            submit_btn = gr.Button("Submit Query")
        with gr.Column(scale=1):
            gr.Markdown("### Node Status")
            node_status = gr.HTML("<div id='node-status'>No nodes running yet.</div>")
            gr.Markdown("### Logs")
            log_output = gr.Textbox(label="Logs", value="", interactive=False, lines=15, elem_id="log-output")
    
    # When the user submits a query, run the full agent process.
    submit_btn.click(
        fn=run_deepgit_interface,
        inputs=[msg],
        outputs=[chatbot, log_output, node_status]
    )
    
    gr.Markdown("### Examples")
    gr.Examples(
        examples=[
            ["Find Python libraries for semantic search in PDFs"],
            ["What are some underrated GitHub tools for time series forecasting?"]
        ],
        inputs=msg
    )

if __name__ == "__main__":
    demo.launch()
