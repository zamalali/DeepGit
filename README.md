<!-- Google Font for pixel-style text -->
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">

<h1 align="center" style="
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 20px;
  font-family: 'Press Start 2P', monospace;
  font-size: 40px;
  color: #111;
  margin-top: 40px;
  margin-bottom: 10px;
">
  <img src="assets/deepgit.png" width="92" height="92" style="display: block;" />
  <span style="
    background: linear-gradient(90deg, #111, #4b0082);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
    letter-spacing: 2px;
  ">
    DeepGit
  </span>
</h1>





<p align="center">
  <img src="assets/flow.png" alt="Workflow Diagram" style="max-width: 800px; width: 100%; height: auto;" />
</p>


## DeepGit

**DeepGit** is an autonomous agent designed to perform deep semantic research across GitHub repositories. It intelligently searches, analyzes, and ranks repositories based on user intent — even for less-known but highly relevant tools.

## ⚙️ How It Works — Agentic Workflow

When a user submits a query, **DeepGit Orchestrator Agent** takes over. Here's the breakdown of the pipeline:

### 🔹 1. Query Expansion Tool
Enhances vague user queries using language models to add specificity and context — enabling more accurate downstream retrieval.

### 🔹 2. Semantic Retrieval Tool
Uses state-of-the-art embedding models to semantically match the enhanced query against a broad set of GitHub repositories.

### 🔹 3. Documentation Intelligence Tool
Summarizes and interprets README files to understand the purpose, setup, and key features of each repository.

### 🔹 4. Codebase Mapping Tool
Analyzes the project’s file structure and technology stack to assess complexity, modularity, and suitability for the user’s needs.

### 🔹 5. Community Insight Tool
Gathers social signals like stars, forks, issues, and pull request activity to gauge real-world engagement and maturity.

### 🔹 6. Relevance Synthesis Tool
Combines insights from all modules to compute a final relevance score tailored to the user query.

### 🔹 7. Insight Delivery Module
Presents ranked repositories to the user with concise summaries and justifications — enabling smart discovery.

## 🚀 Goals

- Surface powerful but under-the-radar open-source tools.
- Build an intelligent layer over GitHub for research-focused developers.
- Open-source the entire workflow to promote transparent research.

---

Want to contribute or give feedback? Reach out or open an issue!

