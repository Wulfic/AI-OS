"""Modularized training CLI commands.

Each command previously lived in `aios.cli.training_cli`. To keep files small
and focused, commands are now split into modules and imported by
`aios.cli.training_cli.register`.
"""

from .train_cmd import train
from .torch_info_cmd import torch_info
from .training_watch_cmd import training_watch
from .english_eval_cmd import english_eval
from .train_ddp_cmd import train_ddp
from .checkpoints_upload_cmd import checkpoints_upload
from .train_multidev_cmd import train_multidev
from .train_parallel_cmd import train_parallel

__all__: list[str] = [
    "train",
    "train_ddp",
    "torch_info",
    "training_watch",
    "english_eval",
    "checkpoints_upload",
    "train_multidev",
    "train_parallel",
]
