from pathlib import Path

import mdtraj
import torch_geometric.data
import torchmetrics
from torchmetrics.utilities import dim_zero_cat

import jamun.utils

from ._utils import get_unique_datasets


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
        print(f"{samples.shape=}")
        return mdtraj.Trajectory(samples.cpu().numpy(), self.topology)


class SaveSampleCallback:
    def __init__(self, datasets: list):
        datasets = sorted(get_unique_datasets(datasets), key=lambda dataset: dataset.label())
        self.meters = {dataset.label(): SampleMetric(topology=dataset.topology) for dataset in datasets}

    def on_sample_start(self, fabric):
        for m in self.meters.values():
            m.to(fabric.device)

    def on_after_sample_batch(self, sample: list, fabric, batch_idx):
        for sample_graph in sample:
            self.meters[sample_graph.dataset_label].update(sample_graph)

    def on_sample_end(self, fabric):
        for label, meter in self.meters.items():
            outdir = Path().resolve() / "samples" / label
            outdir.mkdir(parents=True, exist_ok=True)
            traj = meter.compute()

            if fabric.is_global_zero:
                jamun.utils.save_pdb(traj[0], outdir / "topology.pdb")
                traj.save_dcd(outdir / "traj.dcd")
