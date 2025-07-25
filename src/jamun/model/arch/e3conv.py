from collections.abc import Callable

import e3tools
import torch
import torch_geometric
from e3nn import o3
from e3nn.o3 import Irreps

from jamun.model.noise_conditioning import NoiseConditionalScaling, NoiseConditionalSkipConnection


class E3Conv(torch.nn.Module):
    """A simple E(3)-equivariant convolutional neural network, similar to NequIP."""

    def __init__(
        self,
        irreps_out: str | Irreps,
        irreps_hidden: str | Irreps,
        irreps_sh: str | Irreps,
        atom_embedder_factory: Callable[..., torch.nn.Module],
        hidden_layer_factory: Callable[..., torch.nn.Module],
        output_head_factory: Callable[..., torch.nn.Module],
        radial_edge_embedder_factory: Callable[..., torch.nn.Module],
        bond_edge_embedder_factory: Callable[..., torch.nn.Module],
        n_layers: int,
        max_radius: float = 1.0,  # Default for backward compatibility.
        reduce: str | None = None,
    ):
        super().__init__()

        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.n_layers = n_layers

        self.sh = o3.SphericalHarmonics(irreps_out=self.irreps_sh, normalize=True, normalization="component")
        self.radial_edge_embedder = radial_edge_embedder_factory()
        self.bond_edge_embedder = bond_edge_embedder_factory()
        edge_attr_dim = self.radial_edge_embedder.irreps_out.dim + self.bond_edge_embedder.irreps_out.dim

        self.atom_embedder = atom_embedder_factory()

        self.initial_noise_scaling = NoiseConditionalScaling(self.atom_embedder.irreps_out)
        self.initial_projector = hidden_layer_factory(
            irreps_in=self.initial_noise_scaling.irreps_out,
            irreps_out=self.irreps_hidden,
            irreps_sh=self.irreps_sh,
            edge_attr_dim=edge_attr_dim,
        )

        self.layers = torch.nn.ModuleList()
        self.noise_scalings = torch.nn.ModuleList()
        self.skip_connections = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                hidden_layer_factory(
                    irreps_in=self.irreps_hidden,
                    irreps_out=self.irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    edge_attr_dim=edge_attr_dim,
                )
            )
            self.noise_scalings.append(NoiseConditionalScaling(self.irreps_hidden))
            self.skip_connections.append(NoiseConditionalSkipConnection(self.irreps_hidden))

        self.output_head = output_head_factory(irreps_in=self.irreps_hidden, irreps_out=self.irreps_out)
        self.output_gain = torch.nn.Parameter(torch.tensor(0.0))
        self.reduce = reduce
        self.max_radius = max_radius

    def forward(
        self,
        pos: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        c_noise: torch.Tensor,
        c_in: torch.Tensor,
    ) -> torch.Tensor:
        # Extract edge attributes.
        edge_index = topology["edge_index"]
        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]
        edge_sh = self.sh(edge_vec)

        bonded_edge_attr = self.bond_edge_embedder(topology)
        radial_edge_attr = self.radial_edge_embedder(edge_vec, c_in)
        edge_attr = torch.cat((bonded_edge_attr, radial_edge_attr), dim=-1)

        node_attr = self.atom_embedder(topology)
        node_attr = self.initial_noise_scaling(node_attr, c_noise)
        node_attr = self.initial_projector(node_attr, edge_index, edge_attr, edge_sh)
        for scaling, skip, layer in zip(self.noise_scalings, self.skip_connections, self.layers):
            node_attr = skip(node_attr, layer(scaling(node_attr, c_noise), edge_index, edge_attr, edge_sh), c_noise)
        node_attr = self.output_head(node_attr)
        node_attr = node_attr * self.output_gain

        if self.reduce is not None:
            node_attr = e3tools.scatter(node_attr, batch, dim=0, reduce=self.reduce, dim_size=num_graphs)

        return node_attr
