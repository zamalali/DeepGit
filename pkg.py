import importlib.metadata

# List of packages required for the project.
packages = [
    "requests",
    "numpy",
    "python-dotenv",
    "sentence-transformers",
    "faiss-cpu",
    "pydantic",
    "httpx",
    "gradio",
    "langgraph",
    "langchain_groq",
    "langchain_core",
]

requirements_lines = []

print("Installed package versions:")
for pkg in packages:
    try:
        ver = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        ver = "Not installed"
    line = f"{pkg}=={ver}"
    print(line)
    requirements_lines.append(line)

# Write the package versions to requirements.txt
with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements_lines))

print("\nRequirements have been written to requirements.txt")
