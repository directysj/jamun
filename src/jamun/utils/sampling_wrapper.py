import einops
import torch
import torch.nn as nn
import torch_geometric


class ModelSamplingWrapper:
    """Wrapper to sample positions from a model."""

    def __init__(self, model: nn.Module, init_graphs: torch_geometric.data.Data, sigma: float):
        self._model = model
        self.init_pos = init_graphs.pos
        if init_graphs.get("batch") is None:
            self.batch = torch.zeros(self.init_pos.shape[0], dtype=torch.long, device=self.init_pos.device)
            self.num_graphs = 1
        else:
            self.batch = init_graphs.batch
            self.num_graphs = init_graphs.num_graphs

        self.topology = init_graphs.clone()
        self.sigma = sigma

        del self.topology.pos, self.topology.batch, self.topology.num_graphs

    @property
    def device(self) -> torch.device:
        return self._model.device

    def sample_initial_noisy_positions(self) -> torch.Tensor:
        pos = self.init_pos.clone()
        pos = pos + torch.randn_like(pos) * self.sigma
        return pos

    def __getattr__(self, name: str):
        return getattr(self._model, name)

    def score(self, y: torch.Tensor, sigma: float):
        return self._model.score(
            y,
            topology=self.topology,
            batch=self.batch,
            num_graphs=self.num_graphs,
            sigma=sigma,
        )

    def xhat(self, y: torch.Tensor, sigma: float):
        return self._model.xhat(
            y,
            topology=self.topology,
            batch=self.batch,
            num_graphs=self.num_graphs,
            sigma=sigma,
        )

    def unbatch_samples(self, samples: dict[str, torch.Tensor]) -> list[torch_geometric.data.Data]:
        """Unbatch samples."""

        # Copy off the input graphs, to update attributes later.
        output_graphs = self.topology.clone()
        output_graphs = torch_geometric.data.Batch.to_data_list(output_graphs)

        for key, value in samples.items():
            if value.ndim not in [2, 3]:
                # py_logger = logging.getLogger(__name__)
                # py_logger.info(f"Skipping unbatching of key {key} with shape {value.shape} as it is not 2D or 3D.")
                continue

            if value.ndim == 3:
                value = einops.rearrange(
                    value,
                    "num_frames atoms coords -> atoms num_frames coords",
                )

            unbatched_values = torch_geometric.utils.unbatch(value, self.batch)
            for output_graph, unbatched_value in zip(output_graphs, unbatched_values, strict=True):
                if key in output_graph:
                    raise ValueError(f"Key {key} already exists in the output graph.")

                if unbatched_value.shape[0] != output_graph.num_nodes:
                    raise ValueError(
                        f"Number of nodes in unbatched value ({unbatched_value.shape[0]}) for key {key} does not match "
                        f"number of nodes in output graph ({output_graph.num_nodes})."
                    )

                output_graph[key] = unbatched_value

        return output_graphs
