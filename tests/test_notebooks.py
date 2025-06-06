from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest

# Set notebook directory
NOTEBOOK_DIR = Path("examples")

# Discover all .ipynb files in the directory
notebooks = list(NOTEBOOK_DIR.glob("*.ipynb"))

# Define notebooks to ignore (use Path objects for compatibility)
ignore = {
    NOTEBOOK_DIR / "4a_benchmark.ipynb",
}

# Filter out ignored notebooks
notebooks = [nb for nb in notebooks if nb not in ignore]

@pytest.mark.parametrize("notebook_path", notebooks)
def test_notebook_runs(notebook_path):
    with notebook_path.open(encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Force the correct kernel metadata
    nb.metadata.kernelspec = {
        "name": "python",
        "language": "python",
        "display_name": "Python 3"
    }

    ep = ExecutePreprocessor(timeout=600, kernel_name="python")
    ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
