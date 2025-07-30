import math
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import torch_geometric.data
import torchmetrics
import wandb
from torchmetrics.utilities import dim_zero_cat

import jamun.utils
from jamun.metrics._ramachandran import (
    compute_JS_divergence_of_ramachandran,
    num_dihedrals,
    plot_ramachandran,
)

from ._utils import get_unique_datasets


def log_rama(trajectory, dataset, fabric):
    true_trajectory = dataset.trajectory
    N = num_dihedrals(true_trajectory)

    if fabric.is_global_zero:
        for i in range(N):
            fig, _ = plot_ramachandran(true_trajectory, dihedral_index=i)
            wandb.log({f"{dataset.label()}/ramachandran_static/true_samples/dihedral_{i}": wandb.Image(fig)})
            plt.close(fig)

            fig, _ = plot_ramachandran(trajectory, dihedral_index=i)
            wandb.log({f"{dataset.label()}/ramachandran_static/pred_samples/dihedral_{i}": wandb.Image(fig)})
            plt.close(fig)


def log_js(trajectory, dataset, fabric):
    true_trajectory = dataset.trajectory
    js = compute_JS_divergence_of_ramachandran(trajectory, true_trajectory)
    fabric.log_dict({f"{dataset.label()}/js_divergence": js})


def log_js_divergence_vs_num_samples(trajectory, dataset, fabric, N: int = 21):
    true_trajectory = dataset.trajectory
    n_frames = len(trajectory)
    n_frames_true = len(true_trajectory)

    ref_1 = true_trajectory[: (n_frames_true // 2)]
    ref_2 = true_trajectory[(n_frames_true // 2) :]

    n_samples = [int(x) for x in np.logspace(2, math.log10(n_frames), N)]
    js = [compute_JS_divergence_of_ramachandran(trajectory[:m], ref_2, 100) for m in n_samples]

    n_samples_ref = [int(x) for x in np.logspace(2, math.log10(n_frames_true // 2), N)]
    js_ref = [compute_JS_divergence_of_ramachandran(ref_1[:m], ref_2, 100) for m in n_samples_ref]

    fig, ax = plt.subplots()
    ax.loglog(n_samples, js, marker=".", label="samples")
    ax.loglog(n_samples_ref, js_ref, marker=".", label="ref")
    ax.set_xlabel("n_samples")
    ax.set_ylabel("js divergence")
    ax.set_xmargin(0.0)
    ax.legend()
    fig.tight_layout()

    if fabric.is_global_zero:
        wandb.log({f"{dataset.label()}/js_divergence_vs_n_samples": wandb.Image(fig)})

    plt.close(fig)


class SampleMetric(torchmetrics.Metric):
    def __init__(self, topology: mdtraj.Topology):
        super().__init__()
        self.topology = topology
        self.add_state("samples", default=[], dist_reduce_fx="cat")

    def update(self, data: torch_geometric.data.Batch):
        data = data.to(self.device)
        self.samples.append(data.sample[None, ...])

    def compute(self) -> mdtraj.Trajectory:
        samples = dim_zero_cat(self.samples)
        return mdtraj.Trajectory(samples.cpu().numpy(), self.topology)


class SaveSampleCallback:
    def __init__(self, datasets: list):
        self.datasets = sorted(get_unique_datasets(datasets), key=lambda dataset: dataset.label())
        self.meters = {dataset.label(): SampleMetric(topology=dataset.topology) for dataset in datasets}

    def on_sample_start(self, fabric):
        for m in self.meters.values():
            m.to(fabric.device)

    def on_after_sample_batch(self, sample: list, fabric, batch_idx):
        for sample_graph in sample:
            self.meters[sample_graph.dataset_label].update(sample_graph)

    def on_sample_end(self, fabric):
        for dataset, (label, meter) in zip(self.datasets, self.meters.items()):
            assert dataset.label() == label, f"{dataset.label()=}, {label=}"

            outdir = Path().resolve() / "samples" / label
            outdir.mkdir(parents=True, exist_ok=True)
            traj = meter.compute()

            if fabric.is_global_zero:
                jamun.utils.save_pdb(traj[0], outdir / "topology.pdb")
                traj.save_dcd(outdir / "traj.dcd")

            log_rama(traj, dataset, fabric)
            log_js(traj, dataset, fabric)
            log_js_divergence_vs_num_samples(traj, dataset, fabric)
