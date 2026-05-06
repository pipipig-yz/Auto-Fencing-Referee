"""JSON serialization helpers for numpy scalars and NaN values."""

import json
import math
import numpy as np
from pathlib import Path


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, obj):
        # Replace float nan/inf at Python level too
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return "null"
        return super().encode(obj)


def save_json(data: dict | list, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=_SafeEncoder, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
