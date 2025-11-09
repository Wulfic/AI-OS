from __future__ import annotations

def datasets_stats():
    try:
        from aios.data.datasets import datasets_storage_usage_gb, datasets_storage_cap_gb
        print({"usage_gb": round(float(datasets_storage_usage_gb()), 3), "cap_gb": float(datasets_storage_cap_gb())})
    except Exception as e:
        print({"error": str(e)})
