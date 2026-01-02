# calcium_analysis

This repository contains tools for calcium signal analysis packaged under the `calcium_analysis` module.

Quick install (editable, recommended for development):

This project requires Python 3.10 or newer.

```bash
conda create -n ca_env python=3.10 -y
conda activate ca_env
# ensure pip is available in the environment
pip install -e .
```

After installing, you can:

- Import from Python and notebooks:

```python
import calcium_analysis
from calcium_analysis import peaks
```

- Run the CLI (entry point):

```bash
calcium-analysis --help
python -m calcium_analysis show-modules
```

If you want to use it in Jupyter notebooks inside the same Conda environment, install an IPython kernel for the environment:

```bash
pip install ipykernel
python -m ipykernel install --user --name ca_env --display-name "ca_env"
```

Notes:
- The package code lives under `python/calcium_analysis` so `pyproject.toml` is configured to find packages in the `python` directory.
- Install with `pip install -e .` so edits are picked up inside notebooks and the CLI.
