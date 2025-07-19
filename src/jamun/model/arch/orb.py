import torch
from orb_models.forcefield.angular import SphericalHarmonics
from orb_models.forcefield.base import AtomGraphs
from orb_models.forcefield.gns import MoleculeGNS
from orb_models.forcefield.rbf import BesselBasis

from jamun.model.atom_embedding import AtomEmbeddingWithResidueInformation, SimpleAtomEmbedding


class MoleculeGNSWrapper(torch.nn.Module):
    """A wrapper for the MoleculeGNS model."""

    def __init__(
        self,
        latent_dim: int,
        num_message_passing_steps: int,
        base_mlp_hidden_dim: int,
        base_mlp_depth: int,
        head_mlp_hidden_dim: int,
        head_mlp_depth: int,
        activation: str,
        sh_lmax: int,
        max_radius: float,
        bessel_num_bases: int,
        use_residue_information: bool,
        atom_type_embedding_dim: int,
        atom_code_embedding_dim: int,
        residue_code_embedding_dim: int,
        residue_index_embedding_dim: int,
        bonded_edge_attr_dim: int,
        num_atom_types: int = 20,
        num_atom_codes: int = 10,
        num_residue_types: int = 25,
    ):
        super().__init__()

        if use_residue_information:
            self.atom_embedder = AtomEmbeddingWithResidueInformation(
                atom_type_embedding_dim=atom_type_embedding_dim,
                atom_code_embedding_dim=atom_code_embedding_dim,
                residue_code_embedding_dim=residue_code_embedding_dim,
                residue_index_embedding_dim=residue_index_embedding_dim,
                use_residue_sequence_index=False,
                num_atom_types=num_atom_types,
                max_sequence_length=1,
                num_atom_codes=num_atom_codes,
                num_residue_types=num_residue_types,
            )
        else:
            self.atom_embedder = SimpleAtomEmbedding(
                embedding_dim=atom_type_embedding_dim
                + atom_code_embedding_dim
                + residue_code_embedding_dim,
                max_value=num_atom_types,
            )

        self.bond_edge_embedder = torch.nn.Embedding(2, bonded_edge_attr_dim)

        self.model = MoleculeGNS(
            latent_dim=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            num_mlp_layers=base_mlp_depth,
            mlp_hidden_dim=base_mlp_hidden_dim,
            rbf_transform=BesselBasis(
                r_max=max_radius,
                num_bases=bessel_num_bases,
            ),
            angular_transform=SphericalHarmonics(
                lmax=sh_lmax,
                normalize=True,
                normalization="component",
            ),
            outer_product_with_cutoff=True,
            use_embedding=False,
            expects_atom_type_embedding=False,
            interaction_params={
                "distance_cutoff": True,
                "attention_gate": "sigmoid",
            },
            edge_feature_names=["bond_mask"],
            activation=activation,
            mlp_norm="rms_norm",
        )

    def forward(
        self,
        pos: torch.Tensor,
        topology: AtomGraphs,
        c_noise: torch.Tensor,
        effective_radial_cutoff: float,
    ) -> torch.Tensor:
        topology.pos = pos
        topology.node_features["atomic_numbers_embedding"] = self.atom_embedder(
            topology.node_features["atom_type_index"],
            topology.node_features["atom_code_index"],
            topology.node_features["residue_code_index"],
            topology.node_features["residue_sequence_index"]
        )
        topology.edge_features["bond_mask_embedding"] = self.bond_edge_embedder(
            topology.edge_features["bond_mask"].long()
        )
        return self.model(topology)["pred"]
