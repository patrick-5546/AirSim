import os
from typing import Optional

from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.utils import get_latest_run_id

# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Manually update .venv/lib/python3.8/site-packages/stable_baselines3/common/utils.py
def configure_logger(
    verbose: int = 0,
    tensorboard_log: Optional[str] = None,
    tb_log_name: str = "",
    reset_num_timesteps: bool = True,
) -> Logger:
    """
    Configure the logger's outputs.

    :param verbose: Verbosity level: 0 for no output, 1 for the standard output to be part of the logger outputs
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param tb_log_name: tensorboard log
    :param reset_num_timesteps:  Whether the ``num_timesteps`` attribute is reset or not.
        It allows to continue a previous learning curve (``reset_num_timesteps=False``)
        or start from t=0 (``reset_num_timesteps=True``, the default).
    :return: The logger object
    """
    save_path, format_strings = None, ["stdout"]

    if tensorboard_log is not None and SummaryWriter is None:
        raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")

    if tensorboard_log is not None and SummaryWriter is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
        if verbose >= 1:
            format_strings = ["stdout", "tensorboard", "csv"]
        else:
            format_strings = ["tensorboard", "csv"]
    elif verbose == 0:
        format_strings = [""]
    return configure(save_path, format_strings=format_strings)
