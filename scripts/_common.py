"""Utilidades compartidas por los scripts: ruta del paquete y datos."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bach.config import DATA_DIR  # noqa: E402
from bach.data import build_windows, download, load_all  # noqa: E402


def load_data():
    root = download(DATA_DIR)
    splits = load_all(root)
    return splits, build_windows(splits)
