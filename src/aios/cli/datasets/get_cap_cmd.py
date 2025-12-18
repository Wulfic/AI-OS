from __future__ import annotations


def datasets_get_cap():
    """Get the current dataset storage capacity cap in GB."""
    try:
        from aios.data.datasets import datasets_storage_cap_gb
        print({"cap_gb": float(datasets_storage_cap_gb())})
    except Exception as e:
        print({"error": str(e)})
