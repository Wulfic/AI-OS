from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    # Import modular implementations lazily to keep startup fast and avoid static typing conflicts
    from aios.cli.training.train_cmd import train as _train_cmd
    from aios.cli.training.torch_info_cmd import torch_info as _torch_info_cmd
    from aios.cli.training.training_watch_cmd import training_watch as _training_watch_cmd
    from aios.cli.training.english_eval_cmd import english_eval as _english_eval_cmd
    from aios.cli.training.train_ddp_cmd import train_ddp as _train_ddp_cmd
    from aios.cli.training.checkpoints_upload_cmd import checkpoints_upload as _checkpoints_upload_cmd
    from aios.cli.training.train_multidev_cmd import train_multidev as _train_multidev_cmd
    from aios.cli.training.train_parallel_cmd import train_parallel as _train_parallel_cmd

    app.command("train")(_train_cmd)
    app.command("train-ddp")(_train_ddp_cmd)
    app.command("torch-info")(_torch_info_cmd)
    app.command("training-watch")(_training_watch_cmd)
    app.command("checkpoints-upload")(_checkpoints_upload_cmd)
    app.command("train-multidev")(_train_multidev_cmd)
    app.command("train-parallel")(_train_parallel_cmd)
    app.command("english-eval")(_english_eval_cmd)
