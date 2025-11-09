from __future__ import annotations

import typer


def datasets_list_known(max_size_gb: float = typer.Option(15.0, "--max-size-gb", help="Max dataset size to include")):
    try:
        from aios.data.datasets import known_datasets
        out = []
        for d in known_datasets(max_size_gb):
            out.append({"name": d.name, "url": d.url, "approx_size_gb": d.approx_size_gb, "notes": d.notes})
        print(out)
    except Exception as e:
        print({"error": str(e)})
