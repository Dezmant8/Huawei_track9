"""Reproducibility utilities: seed fixing, device selection, logging setup."""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fix all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"All random seeds set to {seed}")


def get_device() -> str:
    """Select best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"Using CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """Configure logging to console and optionally to file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
