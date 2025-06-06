import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest

# Set notebook directory
NOTEBOOK_DIR = "examples"

# Discover all .ipynb files in the directory
notebooks = [
    os.path.join(NOTEBOOK_DIR, f)
    for f in os.listdir(NOTEBOOK_DIR)
    if f.endswith(".ipynb") and not f.startswith("~$")
]

ignore = [
    "examples/4a_benchmark.ipynb",
]

notebooks = set(notebooks) - set(ignore)

@pytest.mark.parametrize("notebook_path", notebooks)
def test_notebook_runs(notebook_path):
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": NOTEBOOK_DIR}})
