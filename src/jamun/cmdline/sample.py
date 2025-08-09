import logging
import os
import sys
import traceback
from collections.abc import Sequence

import dotenv
import e3nn
import hydra
import lightning.pytorch as pl
import torch
import torch_geometric
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf
from tqdm import tqdm

e3nn.set_optimization_defaults(jit_script_fx=False)

import jamun
import jamun.utils
from jamun.data import MDtrajDataModule, MDtrajDataset
from jamun.hydra import instantiate_dict_cfg
from jamun.hydra.utils import format_resolver
from jamun.utils import dist_log, find_checkpoint

dotenv.load_dotenv(".env", verbose=True)
OmegaConf.register_new_resolver("format", format_resolver)

py_logger = logging.getLogger("jamun")


def sample_loop(
    fabric,
    model,
    sampler,
    num_batches: int,
    init_graphs: torch_geometric.data.Data,
    continue_chain: bool = False,
):
    init_graphs = init_graphs.to(fabric.device)
    model_wrapped = jamun.utils.ModelSamplingWrapper(
        model=model,
        init_graphs=init_graphs,
        sigma=sampler.sigma,
    )

    y_init = model_wrapped.sample_initial_noisy_positions()
    v_init = "gaussian"

    fabric.call("on_sample_start", fabric=fabric)

    batches = range(num_batches)

    if fabric.is_global_zero:
        iterable = tqdm(batches, total=len(batches), desc="Sampling", leave=False)
    else:
        iterable = batches

    with torch.inference_mode():
        for batch_idx in iterable:
            fabric.call("on_before_sample_batch", fabric=fabric, batch_idx=batch_idx)

            out = sampler.sample(model=model_wrapped, y_init=y_init, v_init=v_init)
            samples = model_wrapped.unbatch_samples(out)

            # Start next chain from the end state of the previous chain?
            if continue_chain:
                y_init = out["y"].to(model_wrapped.device)
                v_init = out["v"].to(model_wrapped.device)
            else:
                y_init = model_wrapped.sample_initial_noisy_positions()
                v_init = "gaussian"

            fabric.call("on_after_sample_batch", sample=samples, fabric=fabric, batch_idx=batch_idx)

    fabric.call("on_sample_end", fabric=fabric)


def get_initial_graphs(
    datasets: Sequence[MDtrajDataset], num_init_samples_per_dataset: int, repeat: int = 1
) -> torch_geometric.data.Batch:
    """Get initial graphs for sampling."""
    init_graphs = []
    for dataset in datasets:
        random_indices = torch.randperm(len(dataset))[:num_init_samples_per_dataset]
        for index in random_indices:
            init_graph = dataset[index]
            for _ in range(repeat):
                init_graphs.append(init_graph)
    return torch_geometric.data.Batch.from_data_list(init_graphs)


def run(cfg):
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    rank_zero_logging_level = cfg.get("rank_zero_logging_level", "INFO")
    non_rank_zero_logging_level = cfg.get("non_rank_zero_logging_level", "ERROR")

    if rank_zero_only.rank == 0:
        level = logging.getLevelNamesMapping()[rank_zero_logging_level]
    else:
        level = logging.getLevelNamesMapping()[non_rank_zero_logging_level]

    py_logger.setLevel(level)

    loggers = instantiate_dict_cfg(cfg.get("logger"), verbose=(rank_zero_only.rank == 0))
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, pl.loggers.WandbLogger):
            wandb_logger = logger

    callbacks = instantiate_dict_cfg(cfg.get("callbacks"), verbose=(rank_zero_only.rank == 0))
    fabric = hydra.utils.instantiate(cfg.fabric, callbacks=callbacks, loggers=loggers)

    fabric.launch()

    dist_log(f"{OmegaConf.to_yaml(log_cfg)}")
    dist_log(f"{os.getcwd()=}")
    dist_log(f"{torch.__config__.parallel_info()}")
    if hasattr(os, "sched_getaffinity"):
        dist_log(f"{os.sched_getaffinity(0)=}")

    if matmul_prec := cfg.get("float32_matmul_precision"):
        dist_log(f"Setting float_32_matmul_precision to {matmul_prec}")
        torch.set_float32_matmul_precision(matmul_prec)

    if rank_zero_only.rank == 0 and wandb_logger:
        py_logger.info(f"{wandb_logger.experiment.name=}")
        wandb_logger.experiment.config.update({"cfg": log_cfg, "version": jamun.__version__, "cwd": os.getcwd()})

    # Load the checkpoint either given the wandb run path or the checkpoint path.
    checkpoint_path = find_checkpoint(
        wandb_train_run_path=cfg.get("wandb_train_run_path"),
        checkpoint_dir=cfg.get("checkpoint_dir"),
        checkpoint_type=cfg.get("checkpoint_type"),
    )

    # Overwrite the checkpoint path in the config.
    cfg.model.checkpoint_path = checkpoint_path
    model = hydra.utils.instantiate(cfg.model)

    init_datasets = hydra.utils.instantiate(cfg.init_datasets)
    init_graphs = get_initial_graphs(
        init_datasets,
        num_init_samples_per_dataset=cfg.num_init_samples_per_dataset,
        repeat=cfg.repeat_init_samples,
    )

    fabric.setup(model)
    model.eval()

    sampler = hydra.utils.instantiate(cfg.sampler)

    if seed := cfg.get("seed"):
        # During sampling, we want ranks to generate different chains.
        pl.seed_everything(seed + fabric.global_rank)

    # Run test-time adapation, if specified.
    if finetuning_cfg := cfg.get("finetune_on_init"):
        num_finetuning_steps = finetuning_cfg.get("num_steps")
        dist_log(f"Finetuning for {num_finetuning_steps} steps.")

        # Check that model parameters changed.
        param_sum = sum(p.sum() for p in model.parameters())

        # Train the model for a fixed number of steps.
        trainer = pl.Trainer(
            logger=loggers,
            max_steps=num_finetuning_steps,
            min_steps=num_finetuning_steps,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )
        trainer.fit(
            model,
            datamodule=MDtrajDataModule(
                datasets={"train": init_datasets, "val": init_datasets},
                batch_size=finetuning_cfg.batch_size,
            ),
        )

        # Check that model parameters changed.
        new_param_sum = sum(p.sum() for p in model.parameters())
        dist_log(f"Model parameters changed: {param_sum} -> {new_param_sum}")

    sample_loop(
        fabric=fabric,
        model=model,
        sampler=sampler,
        init_graphs=init_graphs,
        num_batches=cfg.num_batches,
        continue_chain=cfg.continue_chain,
    )

    if wandb_logger:
        wandb_logger.finalize(status="finished")


# Needed for submitit error output.
# See https://github.com/facebookresearch/hydra/issues/2664
@hydra.main(version_base=None, config_path="../hydra_config", config_name="sample")
def main(cfg):
    try:
        run(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
