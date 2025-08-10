import logging

import hydra
from omegaconf import DictConfig


def format_resolver(x: str, pattern: str) -> str:
    return f"{x:{pattern}}"


def instantiate_dict_cfg(cfg: DictConfig | None, verbose: bool = False):
    out = []

    if not cfg:
        return out

    if not isinstance(cfg, DictConfig):
        raise TypeError("cfg must be a DictConfig")

    if verbose:
        py_logger = logging.getLogger(__name__)

    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            if "_target_" in v:
                if verbose:
                    py_logger.info(f"Instantiating <{v._target_}>")
                out.append(hydra.utils.instantiate(v))
            else:
                out.extend(instantiate_dict_cfg(v, verbose=verbose))

    return out
