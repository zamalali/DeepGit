import os
from dotenv import load_dotenv
from pathlib import Path

# Resolve the path to the root directory
dotenv_path = Path(__file__).resolve().parent.parent / ".env"

# Load the .env file
load_dotenv(dotenv_path)

# Example: Access a variable
api_key = os.getenv("GITHUB_API_KEY")
print(f"GITHUB_API_KEY: {api_key}")
