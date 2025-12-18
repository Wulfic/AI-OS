from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    # Lazy import modular implementations to keep startup fast
    from aios.cli.datasets.stats_cmd import datasets_stats as _datasets_stats
    from aios.cli.datasets.get_cap_cmd import datasets_get_cap as _datasets_get_cap
    from aios.cli.datasets.set_cap_cmd import datasets_set_cap as _datasets_set_cap
    from aios.cli.datasets.list_known_cmd import datasets_list_known as _datasets_list_known
    from aios.cli.datasets.build_images_cmd import datasets_build_images as _datasets_build_images
    from aios.cli.datasets.build_text_cmd import datasets_build_text as _datasets_build_text
    from aios.cli.datasets.build_videos_cmd import datasets_build_videos as _datasets_build_videos
    from aios.cli.datasets.build_websites_cmd import datasets_build_websites as _datasets_build_websites
    from aios.cli.datasets.build_raw_cmd import datasets_build_raw as _datasets_build_raw

    app.command("datasets-stats")(_datasets_stats)
    app.command("datasets-get-cap")(_datasets_get_cap)
    app.command("datasets-set-cap")(_datasets_set_cap)
    app.command("datasets-list-known")(_datasets_list_known)
    app.command("datasets-build-images")(_datasets_build_images)
    app.command("datasets-build-videos")(_datasets_build_videos)
    app.command("datasets-build-text")(_datasets_build_text)
    app.command("datasets-build-websites")(_datasets_build_websites)
    app.command("datasets-build-raw")(_datasets_build_raw)
