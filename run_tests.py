import sys
import os
import pytest

if __name__ == "__main__":
    # Set the project root (assuming agent.py, tools, and tests are in the "DeepGit" folder)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print("Project root added to sys.path:", project_root)
    
    # Define the tests directory (inside the DeepGit folder)
    tests_dir = os.path.join(project_root, "tests")
    print("Running tests from:", tests_dir)
    
    # Run pytest on the tests folder
    result = pytest.main([tests_dir])
    
    if result == 0:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    sys.exit(result)
