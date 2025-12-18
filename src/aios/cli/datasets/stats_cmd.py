from __future__ import annotations
import json

def datasets_stats():
    try:
        from aios.data.datasets import datasets_storage_usage_gb, datasets_storage_cap_gb
        result = {"usage_gb": round(float(datasets_storage_usage_gb()), 3), "cap_gb": float(datasets_storage_cap_gb())}
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
