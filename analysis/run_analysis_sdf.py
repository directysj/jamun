import logging
import sys
import tempfile
import warnings
from typing import Any

import mdtraj as md
import numpy as np
import openbabel
import pyemma
import pyemma.coordinates.clustering
import pyemma.coordinates.data
from openbabel import pybel
from rdkit import Chem
from scipy.spatial import distance

sys.path.append("./")
import run_analysis

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("analysis_sdf")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pyemma.util.exceptions.PyEMMA_DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_smiles_from_sdf(file_name: str) -> dict[str, str]:
    """Extract SMILES strings from an SDF file using RDKit."""
    sppl = Chem.SDMolSupplier(file_name)
    smiles_dict = {}
    for mol in sppl:
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name")
            smiles_dict[name] = smi
    return smiles_dict


def convert_smiles_to_pdb(smiles_str: str, output_pdb_file: str) -> None:
    """Convert a SMILES string to a PDB file using Open Babel."""
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "pdb")
    obMol = openbabel.OBMol()
    obConversion.ReadString(obMol, smiles_str)
    obConversion.WriteFile(obMol, output_pdb_file)


def convert_sdf_conformers_to_pdb(sdf_file: str, output_pdb_file: str):
    # Load the SDF file containing multiple conformers
    sdf_molecule = list(pybel.readfile("sdf", sdf_file))

    # Write each conformer as separate MODEL/ENDMDL sections in the PDB file
    with open(output_pdb_file, "w") as f:
        for model_index, mol in enumerate(sdf_molecule):
            mol.OBMol.DeleteHydrogens()
            f.write(f"MODEL     {model_index + 1}\n")
            f.write(mol.write("pdb"))
            f.write("ENDMDL\n")


def featurize_trajectory_with_bb_sc_pdb(traj: md.Trajectory, cossin: bool) -> dict:
    """Featurize an MDTraj trajectory with backbone, and sidechain torsion angles using pyEMMA."""

    # Step 1 - Extract backbone atom numbers from the temporary PDB file
    backbone_atoms = {"CA": [], "C": [], "N": [], "O": []}
    sidechain_atoms = []  # List to store sidechain atom indices

    for atom in traj.topology.atoms:
        atom_name = atom.name
        residue_name = atom.residue.name
        atom_index = atom.index

        # Classify backbone atoms
        if atom_name in backbone_atoms:
            backbone_atoms[atom_name].append(atom_index)
        # Classify sidechain atoms (exclude backbone atoms)
        elif residue_name not in ["HOH"]:  # Exclude water or other non-amino acid residues
            sidechain_atoms.append(atom_index)

    # Ensure all backbone atoms are found
    for atom_type, atom_list in backbone_atoms.items():
        if not atom_list:
            raise ValueError(f"Backbone atom '{atom_type}' not found in the trajectory topology.")

    if not sidechain_atoms:
        raise ValueError("No sidechain atoms found in the trajectory topology.")

    feats = pyemma.coordinates.featurizer(traj.topology)
    py_logger.info("topology from bb_sc_pdb:", traj.topology)

    # Backbone torsion angles
    [
        [
            backbone_atoms["CA"][i],
            backbone_atoms["C"][i],
            backbone_atoms["N"][(i + 1) % len(backbone_atoms["N"])],
            backbone_atoms["CA"][(i + 1) % len(backbone_atoms["CA"])],
        ]
        for i in range(len(backbone_atoms["CA"]))
    ]

    phi_indices = [
        [
            backbone_atoms["C"][i],
            backbone_atoms["N"][i],
            backbone_atoms["CA"][i],
            backbone_atoms["C"][(i + 1) % len(backbone_atoms["C"])],
        ]
        for i in range(len(backbone_atoms["N"]))
    ]

    psi_indices = [
        [
            backbone_atoms["N"][i],
            backbone_atoms["CA"][i],
            backbone_atoms["C"][i],
            backbone_atoms["N"][(i + 1) % len(backbone_atoms["N"])],
        ]
        for i in range(len(backbone_atoms["CA"]))
    ]
    feats.add_dihedrals(phi_indices, cossin=cossin)
    feats.add_dihedrals(psi_indices, cossin=cossin)

    # Sidechain torsion angles
    sidechain_torsion_indices = [sidechain_atoms[i : i + 4] for i in range(len(sidechain_atoms) - 3)]
    feats.add_dihedrals(sidechain_torsion_indices, cossin=cossin)
    # Map atom indices to feature indices
    phi_feature_indices = list(range(len(phi_indices)))
    psi_feature_indices = list(range(len(phi_indices), len(phi_indices) + len(psi_indices)))

    traj_featurized = feats.transform(traj)

    return feats, traj_featurized, phi_feature_indices, psi_feature_indices


def featurize_trajectory_with_CA_distances(traj: md.Trajectory, temp_pdb_file: str) -> dict:
    # Step 1 - Extract backbone atom numbers from the temporary PDB file
    ca_indices = []

    with open(temp_pdb_file) as pdb:
        for line in pdb:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                atom_number = int(line[6:11].strip()) - 1  # MDTraj uses 0-based indexing
                if atom_name == "CA":
                    ca_indices.append(atom_number)

    if not ca_indices:
        raise ValueError("No CA atoms found in the temporary PDB file.")

    # Step 2 - Featurize the trajectory using PyEMMA backbone atoms
    feats = pyemma.coordinates.featurizer(traj.topology)
    ca_pairs = [(i, j) for i in ca_indices for j in ca_indices if i < j]  # Generate all unique CA pairs
    feats.add_distances(ca_pairs, periodic=False)

    traj_featurized = feats.transform(traj)
    return feats, traj_featurized


# Function to featurize a trajectory using backbone atom numbers
def featurize_trajectory_with_backbone_atoms(traj: md.Trajectory, temp_pdb_file: str, cossin: bool) -> dict:
    # Step 1 - Extract backbone atom numbers from the temporary PDB file
    backbone_atoms = {"CA": [], "C": [], "N": [], "O": []}
    sidechain_atoms = []

    with open(temp_pdb_file) as pdb:
        for line in pdb:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                atom_number = int(line[6:11].strip()) - 1  # MDTraj uses 0-based indexing
                if atom_name in backbone_atoms:
                    backbone_atoms[atom_name].append(atom_number)

    for atom_type, atom_list in backbone_atoms.items():
        if not atom_list:
            raise ValueError(f"Backbone atom '{atom_type}' not found in the temporary PDB file.")

    # Step 2 - Featurize the trajectory using PyEMMA backbone atoms
    feats = pyemma.coordinates.featurizer(traj.topology)
    # Backbone torsion angles
    [
        [
            backbone_atoms["CA"][i],
            backbone_atoms["C"][i],
            backbone_atoms["N"][(i + 1) % len(backbone_atoms["N"])],
            backbone_atoms["CA"][(i + 1) % len(backbone_atoms["CA"])],
        ]
        for i in range(len(backbone_atoms["CA"]))
    ]

    phi_indices = [
        [
            backbone_atoms["C"][i],
            backbone_atoms["N"][i],
            backbone_atoms["CA"][i],
            backbone_atoms["C"][(i + 1) % len(backbone_atoms["C"])],
        ]
        for i in range(len(backbone_atoms["N"]))
    ]

    psi_indices = [
        [
            backbone_atoms["N"][i],
            backbone_atoms["CA"][i],
            backbone_atoms["C"][i],
            backbone_atoms["N"][(i + 1) % len(backbone_atoms["N"])],
        ]
        for i in range(len(backbone_atoms["CA"]))
    ]
    feats.add_dihedrals(phi_indices, cossin=cossin)
    feats.add_dihedrals(psi_indices, cossin=cossin)

    # Sidechain torsion angles
    sidechain_torsion_indices = [sidechain_atoms[i : i + 4] for i in range(len(sidechain_atoms) - 3)]
    feats.add_dihedrals(sidechain_torsion_indices, cossin=cossin)
    # Map atom indices to feature indices
    phi_feature_indices = list(range(len(phi_indices)))
    psi_feature_indices = list(range(len(phi_indices), len(phi_indices) + len(psi_indices)))

    traj_featurized = feats.transform(traj)

    return feats, traj_featurized, phi_feature_indices, psi_feature_indices


def featurize_trajectories_macrocycle(traj_md: md.Trajectory, sdf_file: str) -> dict[str, dict[str, np.ndarray]]:
    """Featurize MDTraj trajectories with backbone, and sidechain torsion angles and distances using pyEMMA."""
    return {
        "traj": load_and_featurize_trajectories(traj_md, sdf_file),
        "ref_traj": load_and_featurize_trajectories_sdf(sdf_file),
    }


def extract_atom_names_from_pdb(pdb_file: str) -> list:
    """Extract atom names from a PDB file."""
    # Load the PDB file using MDTraj
    pdb_traj = md.load(pdb_file)
    return [atom.name for atom in pdb_traj.topology.atoms]


def rename_atoms_in_topology(topology: md.Topology, atom_names: list) -> md.Topology:
    """Rename atoms in the topology based on provided atom names."""
    for atom, new_name in zip(topology.atoms, atom_names):
        atom.name = new_name
    return topology


# Function to load and featurize trajectories
def load_and_featurize_trajectories(traj_md: md.Trajectory, sdf_file: str) -> dict[str, dict[str, np.ndarray]]:
    """Covert SDF to PDB and featurize the trajectory with backbone, and sidechain torsion angles and distances using pyEMMA."""

    # Convert SDF to PDB and extract atom names
    pdb_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb").name
    convert_sdf_conformers_to_pdb(sdf_file, pdb_file)
    atom_names = extract_atom_names_from_pdb(pdb_file)

    # Ensure the number of atoms match
    if len(atom_names) != traj_md.n_atoms:
        raise ValueError("Number of atoms in PDB does not match number of atoms in DCD trajectory.")

    # Rename atoms in the trajectory
    traj_md.topology = rename_atoms_in_topology(traj_md.topology, atom_names)

    """Featurize an MDTraj trajectory with backbone, and sidechain torsion angles and distances using pyEMMA."""
    feats, traj_featurized, phi_indices, psi_indices = featurize_trajectory_with_bb_sc_pdb(traj_md, cossin=False)
    feats_cossin, traj_featurized_cossin, phi_indices, psi_indices = featurize_trajectory_with_bb_sc_pdb(
        traj_md, cossin=True
    )
    feats_dists, traj_featurized_dists = run_analysis.featurize_trajectory_with_distances(traj_md)
    return {
        "feats": {
            "torsions": feats,
            "torsions_cossin": feats_cossin,
            "distances": feats_dists,
            "phi_indices": phi_indices,
            "psi_indices": psi_indices,
        },
        "traj_featurized": {
            "torsions": traj_featurized,
            "torsions_cossin": traj_featurized_cossin,
            "distances": traj_featurized_dists,
            "phi_indices": phi_indices,
            "psi_indices": psi_indices,
        },
    }


def load_and_featurize_trajectories_sdf(sdf_file: str) -> dict[str, dict[str, np.ndarray]]:
    """Covert SDF to PDB and featurize the trajectory with backbone, and sidechain torsion angles and distances using pyEMMA."""
    pdb_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb").name
    convert_sdf_conformers_to_pdb(sdf_file, pdb_file)
    traj = md.load_pdb(pdb_file)

    if not isinstance(traj, md.Trajectory):
        raise TypeError(f"Expected traj to be md.Trajectory, but got {type(traj)}")

    """Featurize an MDTraj trajectory with backbone, and sidechain torsion angles and distances using pyEMMA."""
    feats, traj_featurized, phi_indices, psi_indices = featurize_trajectory_with_bb_sc_pdb(traj, cossin=False)
    feats_cossin, traj_featurized_cossin, phi_indices, psi_indices = featurize_trajectory_with_bb_sc_pdb(
        traj, cossin=True
    )
    feats_dists, traj_featurized_dists = run_analysis.featurize_trajectory_with_distances(traj)

    return {
        "feats": {
            "torsions": feats,
            "torsions_cossin": feats_cossin,
            "distances": feats_dists,
            "phi_indices": phi_indices,
            "psi_indices": psi_indices,
        },
        "traj_featurized": {
            "torsions": traj_featurized,
            "torsions_cossin": traj_featurized_cossin,
            "distances": traj_featurized_dists,
            "phi_indices": phi_indices,
            "psi_indices": psi_indices,
        },
    }


def compute_PMF_macrocycle(
    traj_featurized: np.ndarray,
    phi_indices: list[int],
    psi_indices: list[int],
    ref_traj: np.ndarray,
    num_bins: int = 50,
    internal_angles: bool = False,
) -> dict[str, np.ndarray]:
    if internal_angles:
        psi_indices = psi_indices[1:]
        phi_indices = phi_indices[:-1]

    phi = traj_featurized[:, phi_indices]
    psi = traj_featurized[:, psi_indices]

    num_dihedrals = phi.shape[1]

    pmf = np.zeros((num_dihedrals, num_bins - 1, num_bins - 1))
    xedges = np.linspace(-np.pi, np.pi, num_bins)
    yedges = np.linspace(-np.pi, np.pi, num_bins)

    for dihedral_index in range(num_dihedrals):
        H, _, _ = np.histogram2d(
            phi[:, dihedral_index], psi[:, dihedral_index], bins=np.linspace(-np.pi, np.pi, num_bins)
        )
        pmf[dihedral_index] = -np.log(H.T) + np.max(np.log(H.T))

    return {
        "pmf": pmf,
        "xedges": xedges,
        "yedges": yedges,
    }


def compute_PMFs_macrocycle(
    traj: np.ndarray, phi_indices, psi_indices, ref_traj: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> dict[str, np.ndarray]:
    """Compute the potential of mean force (PMF) for a trajectory along a dihedral angle."""
    return {
        "traj": {
            "pmf_all": compute_PMF_macrocycle(traj, phi_indices, psi_indices, feats, internal_angles=False),
            # "pmf_internal": compute_PMF_macrocycle(traj, phi_indices, psi_indices, feats,  internal_angles=True),
        },
        "ref_traj": {
            "pmf_all": compute_PMF_macrocycle(ref_traj, phi_indices, psi_indices, feats, internal_angles=False),
            # "pmf_internal": compute_PMF_macrocycle(ref_traj,phi_indices, psi_indices, feats, internal_angles=True),
        },
    }


def compute_JSD_torsion_stats_macrocycle(
    traj_featurized: np.ndarray,
    ref_traj_featurized: np.ndarray,
    phi_indices: list[int],
    psi_indices: list[int],
    feats: pyemma.coordinates.data.MDFeaturizer,
    chi_indices: list[int] | None,  # Add sidechain torsion indices
) -> dict[str, float]:
    """Compute Jenson-Shannon distances for a trajectory and reference trajectory. Taken from MDGen."""
    results = {}
    for i, feat in enumerate(feats.describe()):
        ref_p = np.histogram(ref_traj_featurized[:, i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(traj_featurized[:, i], range=(-np.pi, np.pi), bins=100)[0]
        results[feat] = distance.jensenshannon(ref_p, traj_p)

    # Compute JSDs for backbone, sidechain, and all torsions using provided indices.
    results["backbone_torsions"] = np.mean(
        [
            distance.jensenshannon(
                np.histogram(ref_traj_featurized[:, idx], range=(-np.pi, np.pi), bins=100)[0],
                np.histogram(traj_featurized[:, idx], range=(-np.pi, np.pi), bins=100)[0],
            )
            for idx in phi_indices + psi_indices
        ]
    )
    # Compute JSDs for sidechain torsions (chi)
    if chi_indices:
        results["sidechain_torsions"] = np.mean(
            [
                distance.jensenshannon(
                    np.histogram(ref_traj_featurized[:, idx], range=(-np.pi, np.pi), bins=100)[0],
                    np.histogram(traj_featurized[:, idx], range=(-np.pi, np.pi), bins=100)[0],
                )
                for idx in chi_indices
            ]
        )
    else:
        results["sidechain_torsions"] = np.nan  # If no sidechain torsions are provided
    # Compute JSDs for all torsions (backbone + sidechain)
    if chi_indices:
        all_indices = phi_indices + psi_indices + chi_indices
    else:
        all_indices = phi_indices + psi_indices

    results["all_torsions"] = np.mean(
        [
            distance.jensenshannon(
                np.histogram(ref_traj_featurized[:, idx], range=(-np.pi, np.pi), bins=100)[0],
                np.histogram(traj_featurized[:, idx], range=(-np.pi, np.pi), bins=100)[0],
            )
            for idx in all_indices
        ]
    )

    # Use the provided phi and psi indices for pairwise calculations.
    for phi_index, psi_index in zip(phi_indices, psi_indices):
        ref_features = np.stack([ref_traj_featurized[:, phi_index], ref_traj_featurized[:, psi_index]], axis=1)
        ref_p = np.histogram2d(*ref_features.T, range=((-np.pi, np.pi), (-np.pi, np.pi)), bins=50)[0]

        traj_features = np.stack([traj_featurized[:, phi_index], traj_featurized[:, psi_index]], axis=1)
        traj_p = np.histogram2d(*traj_features.T, range=((-np.pi, np.pi), (-np.pi, np.pi)), bins=50)[0]

        phi_psi_feats = [feats.describe()[phi_index], feats.describe()[psi_index]]
        results["|".join(phi_psi_feats)] = distance.jensenshannon(ref_p.flatten(), traj_p.flatten())

    return results


def compute_JSD_torsion_stats_against_time_for_trajectory_macrocycle(
    traj_featurized: np.ndarray,
    ref_traj_featurized: np.ndarray,
    phi_indices,
    psi_indices,
    feats: pyemma.coordinates.data.MDFeaturizer,
    chi_indices: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Computes the Jenson-Shannon distance between the Ramachandran distributions of a trajectory and a reference trajectory at different time points."""
    steps = np.logspace(0, np.log10(len(traj_featurized)), num=10, dtype=int)
    return {
        step: compute_JSD_torsion_stats_macrocycle(
            traj_featurized[:step],
            ref_traj_featurized,
            phi_indices,
            psi_indices,
            feats,
            chi_indices,
        )
        for step in steps
    }


def compute_JSD_torsion_stats_against_time_macrocycle(
    traj_featurized: np.ndarray,
    ref_traj_featurized: np.ndarray,
    phi_indices,
    psi_indices,
    feats: pyemma.coordinates.data.MDFeaturizer,
    chi_indices: list[int] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """Computes the Jenson-Shannon distance between the Ramachandran distributions of a trajectory and a reference trajectory at different time points."""
    return {
        "traj": compute_JSD_torsion_stats_against_time_for_trajectory_macrocycle(
            traj_featurized, ref_traj_featurized, phi_indices, psi_indices, feats
        ),
        "ref_traj": compute_JSD_torsion_stats_against_time_for_trajectory_macrocycle(
            ref_traj_featurized, ref_traj_featurized, phi_indices, psi_indices, feats
        ),
    }


def analyze_trajectories_sdf(traj_md: md.Trajectory, sdf_file: str) -> dict[str, Any]:
    """Run analysis on the trajectories and return results dictionary."""

    # Featurize trajectories.
    results = {}
    results["featurization"] = featurize_trajectories_macrocycle(traj_md, sdf_file)

    traj_results = results["featurization"]["traj"]
    traj_feats = traj_results["feats"]["torsions"]
    traj_featurized_dict = traj_results["traj_featurized"]
    traj_featurized = traj_featurized_dict["torsions"]

    ref_traj_results = results["featurization"]["ref_traj"]
    ref_traj_featurized_dict = ref_traj_results["traj_featurized"]
    ref_traj_featurized = ref_traj_featurized_dict["torsions"]
    py_logger.info("Featurization complete.")

    py_logger.info(f"traj results: {traj_results}")
    py_logger.info(f"ref traj results: {ref_traj_results}")
    py_logger.info(f"traj feats: {traj_feats}")
    py_logger.info(f"ref traj feats: {ref_traj_results['feats']}")
    py_logger.info(f"traj featurized dict: {traj_featurized_dict}")
    py_logger.info(f"ref traj featurized dict: {ref_traj_featurized_dict}")
    py_logger.info(f"traj featurized: {traj_featurized}")
    py_logger.info(f"ref traj featurized: {ref_traj_featurized}")

    # Compute feature histograms.
    results["feature_histograms"] = run_analysis.compute_feature_histograms(
        traj_featurized_dict,
        ref_traj_featurized_dict,
    )
    py_logger.info("Feature histograms computed.")

    # Compute PMFs.
    results["PMFs"] = compute_PMFs_macrocycle(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info("PMFs computed.")

    # Compute JSDs.
    results["JSD_torsion_stats"] = run_analysis.compute_JSD_torsion_stats(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info("JSD torsion stats computed.")

    # Compute JSDs of torsions against time.
    results["JSD_torsion_stats_against_time"] = run_analysis.compute_JSD_torsion_stats_against_time(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info("JSD torsion stats as a function of time computed.")

    traj_featurized_cossin = traj_featurized_dict["torsions_cossin"]
    ref_traj_featurized_cossin = ref_traj_featurized_dict["torsions_cossin"]

    # TICA analysis.
    results["TICA"] = run_analysis.compute_TICA(
        traj_featurized_cossin,
        ref_traj_featurized_cossin,
    )
    py_logger.info("TICA computed.")

    traj_tica = results["TICA"]["traj_tica"]
    ref_traj_tica = results["TICA"]["ref_traj_tica"]

    # Compute TICA stats.
    results["TICA_stats"] = run_analysis.compute_TICA_stats(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info("TICA stats computed.")

    # Compute autocorrelation stats.
    results["autocorrelation_stats"] = run_analysis.compute_autocorrelation_stats(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info("Autocorrelation stats computed.")

    # Compute MSM stats.
    # Sometimes, this fails because the reference trajectory is too short.
    try:
        results["MSM_stats"] = run_analysis.compute_MSM_stats(
            traj_tica,
            ref_traj_tica,
        )
        py_logger.info("MSM stats computed.")

        # Compute JSDs against time.
        results["JSD_MSM_stats_against_time"] = run_analysis.compute_JSD_MSM_stats_against_time(
            traj_tica,
            ref_traj_tica,
        )
        py_logger.info("JSD MSM stats as a function of time computed.")

    except IndexError:
        py_logger.warning("MSM stats could not be computed.")

    return results
