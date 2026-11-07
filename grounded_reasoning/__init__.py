"""Root directories, overridable with environment variables.

GR_DATA     datasets, benchmark images and generated training data   (default: <repo>/data)
GR_RUNS     model checkpoints                                         (default: <repo>/runs)
GR_RESULTS  evaluation outputs, tables and figures                    (default: <repo>/results)
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("GR_DATA", REPO / "data"))
RUNS = Path(os.environ.get("GR_RUNS", REPO / "runs"))
RESULTS = Path(os.environ.get("GR_RESULTS", REPO / "results"))
