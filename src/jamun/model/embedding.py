import e3nn
import torch
import torch.nn as nn
import torch_geometric


class SimpleAtomEmbedding(nn.Module):
    """Embed atoms without residue information."""

    def __init__(self, embedding_dim: int, max_value: int):
        super().__init__()
        self.embedding = nn.Embedding(max_value, embedding_dim)
        self.irreps_out = e3nn.o3.Irreps(f"{embedding_dim}x0e")

    def forward(self, topology: torch_geometric.data.Data) -> torch.Tensor:
        return self.embedding(topology["atom_type_index"])


class AtomEmbeddingWithResidueInformation(nn.Module):
    """Embed atoms with residue information."""

    def __init__(
        self,
        atom_type_embedding_dim: int,
        atom_code_embedding_dim: int,
        residue_code_embedding_dim: int,
        residue_index_embedding_dim: int,
        use_residue_sequence_index: bool,
        num_atom_types: int,
        max_sequence_length: int,
        num_atom_codes: int,
        num_residue_types: int,
        input_as_atom_graphs: bool = False,
    ):
        super().__init__()
        self.atom_type_embedding = torch.nn.Embedding(num_atom_types, atom_type_embedding_dim)
        self.atom_code_embedding = torch.nn.Embedding(num_atom_codes, atom_code_embedding_dim)
        self.residue_code_embedding = torch.nn.Embedding(num_residue_types, residue_code_embedding_dim)
        self.residue_index_embedding = torch.nn.Embedding(max_sequence_length, residue_index_embedding_dim)

        self.use_residue_sequence_index = use_residue_sequence_index
        self.irreps_out = e3nn.o3.Irreps(
            f"{atom_type_embedding_dim}x0e + {atom_type_embedding_dim}x0e + {residue_code_embedding_dim}x0e + {residue_index_embedding_dim}x0e"
        )
        self.input_as_atom_graphs = input_as_atom_graphs

    def forward(
        self,
        topology: torch_geometric.data.Data,
    ) -> torch.Tensor:
        if self.input_as_atom_graphs:
            topology = topology.node_features

        features = []
        atom_type_embedded = self.atom_type_embedding(topology["atom_type_index"])
        features.append(atom_type_embedded)

        atom_code_embedded = self.atom_code_embedding(topology["atom_code_index"])
        features.append(atom_code_embedded)

        residue_code_embedded = self.residue_code_embedding(topology["residue_code_index"])
        features.append(residue_code_embedded)

        if not self.use_residue_sequence_index:
            torch.zeros_like(topology["atom_type_index"])

        residue_sequence_index_embedded = self.residue_index_embedding(topology["residue_sequence_index"])
        features.append(residue_sequence_index_embedded)

        features = torch.cat(features, dim=-1)
        return features


class BondEdgeEmbedder(nn.Module):
    """Embed bond types."""

    def __init__(self, bonded_edge_attr_dim: int, input_as_atom_graphs: bool = False):
        super().__init__()
        self.bonded_edge_attr_dim = bonded_edge_attr_dim
        self.bond_edge_embedder = torch.nn.Embedding(2, bonded_edge_attr_dim)
        self.input_as_atom_graphs = input_as_atom_graphs
        self.irreps_out = e3nn.o3.Irreps(f"{bonded_edge_attr_dim}x0e")

    def forward(self, topology: torch_geometric.data.Data) -> torch.Tensor:
        if self.input_as_atom_graphs:
            topology = topology.edge_features
        return self.bond_edge_embedder(topology["bond_mask"])


class RadialEdgeEmbedder(nn.Module):
    """Embed radial edge attributes."""

    def __init__(self, radial_edge_attr_dim: int, max_radius: float, basis: str, cutoff: bool):
        super().__init__()
        self.radial_edge_attr_dim = radial_edge_attr_dim
        self.max_radius = max_radius
        self.basis = basis
        self.cutoff = cutoff
        self.irreps_out = e3nn.o3.Irreps(f"{radial_edge_attr_dim}x0e")

    def forward(self, edge_vec: torch.Tensor, c_in: torch.Tensor) -> torch.Tensor:
        return e3nn.math.soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            0.0,
            (c_in * self.max_radius).squeeze(-1),
            self.radial_edge_attr_dim,
            basis=self.basis,
            cutoff=self.cutoff,
        )
