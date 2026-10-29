# ML Systems Labs

Labs based on Reddi, *Introduction to Machine Learning Systems*.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Each lab is run from the repository root:

| Lab | Topic | Command |
|-----|-------|---------|
| 01 | Environment and first system measurements | `python lab01/src/run_all.py` |
| 02 | Neural network from scratch | `python lab02/src/run_all.py` |
| 03 | Architectures and their costs | `python lab03/src/run_all.py` |
| 04 | Data pipeline and quality | `python lab04/src/pipeline.py` |

Datasets are downloaded into `data/` automatically (ignored by Git).
Results are written into each lab's `results/` folder, and each lab has a `report.md`.
