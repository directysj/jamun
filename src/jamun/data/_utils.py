import collections
import concurrent.futures
import os
import re
from collections.abc import Callable, Iterable, Sequence

import hydra
import pandas as pd
import requests
import torch
from tqdm.auto import tqdm

from jamun.data._mdtraj import MDtrajDataset, MDtrajIterableDataset
from jamun.data._sdf import MDtrajSDFDataset


def dloader_map_reduce(
    f: Callable[[torch.Tensor], torch.Tensor],
    dloader: Iterable[torch.Tensor],
    reduce_fn: Callable[[list[torch.Tensor]], torch.Tensor] = torch.cat,
    verbose: bool = False,
):
    """Map a function over a data loader and reduce the results."""
    outs = []
    for batch in tqdm(dloader, disable=not verbose):
        outs.append(f(batch))
    return reduce_fn(outs)


def dataset_factory_wrapper(x):
    dataset_class, kwargs = x
    return dataset_class(**kwargs)


def download_file(url: str, path: str, verbose: bool = False, block_size: int | None = None):
    """Download a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    if block_size is None:
        block_size = 1024 * 1024

    with open(path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, disable=not verbose) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))


def parse_datasets_from_directory(
    root: str,
    traj_pattern: str,
    pdb_pattern: str | None = None,
    pdb_file: Sequence[str] | None = None,
    max_datasets: int | None = None,
    max_datasets_offset: int | None = None,
    filter_codes: Sequence[str] | None = None,
    as_iterable: bool = False,
    num_workers: int = 0,
    **dataset_kwargs,
) -> list[MDtrajDataset]:
    """Helper function to create MDtrajDataset objects from a directory of trajectory files."""
    if pdb_file is not None and pdb_pattern is not None:
        raise ValueError("Exactly one of pdb_file and pdb_pattern should be provided.")

    traj_prefix, traj_pattern = os.path.split(traj_pattern)
    traj_pattern_compiled = re.compile(traj_pattern)
    if "*" in traj_prefix or "?" in traj_prefix:
        raise ValueError("traj_prefix should not contain wildcards.")

    traj_files = collections.defaultdict(list)
    codes = set()
    for entry in os.scandir(os.path.join(root, traj_prefix)):
        match = traj_pattern_compiled.match(entry.name)
        if not match:
            continue

        code = match.group(1)
        codes.add(code)
        traj_files[code].append(os.path.join(traj_prefix, entry.name))

    if len(codes) == 0:
        raise ValueError("No codes found in directory.")

    pdb_files = {}
    if pdb_pattern is not None:
        pdb_prefix, pdb_pattern = os.path.split(pdb_pattern)
        pdb_pattern_compiled = re.compile(pdb_pattern)
        if "*" in pdb_prefix or "?" in pdb_prefix:
            raise ValueError("pdb_prefix should not contain wildcards.")

        for entry in os.scandir(os.path.join(root, pdb_prefix)):
            match = pdb_pattern_compiled.match(entry.name)
            if not match:
                continue

            code = match.group(1)
            if code not in codes:
                continue
            pdb_files[code] = os.path.join(pdb_prefix, entry.name)
    else:
        for code in codes:
            pdb_files[code] = pdb_file

    # Filter out codes.
    if filter_codes is not None:
        codes = [code for code in codes if code in set(filter_codes)]

    # Sort the codes and offset them, if necessary.
    codes = list(sorted(codes))
    if max_datasets_offset is not None:
        codes = codes[max_datasets_offset:]
    if max_datasets is not None:
        codes = codes[:max_datasets]

    if as_iterable:
        dataset_class = MDtrajIterableDataset
    else:
        dataset_class = MDtrajDataset

    args_list = [
        (
            dataset_class,
            {
                "root": root,
                "traj_files": tuple(traj_files[code]),
                "pdb_file": pdb_files[code],
                "label": code,
                **dataset_kwargs,
            },
        )
        for code in codes
    ]

    if num_workers > 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            datasets = list(
                tqdm(executor.map(dataset_factory_wrapper, args_list), total=len(args_list), desc="Creating datasets")
            )
    else:
        datasets = list(tqdm(map(dataset_factory_wrapper, args_list), total=len(args_list), desc="Creating datasets"))

    return datasets


def parse_datasets_from_directory_new(
    root: str,
    traj_pattern: str,
    topology_pattern: str | None = None,
    topology_file: Sequence[str] | None = None,
    max_datasets: int | None = None,
    max_datasets_offset: int | None = None,
    filter_codes: Sequence[str] | None = None,
    filter_codes_csv: str | None = None,
    filter_codes_csv_header: str | None = None,
    as_iterable: bool = False,
    as_sdf: bool = False,
    **dataset_kwargs,
) -> list[MDtrajDataset]:
    """Helper function to create MDtrajDataset objects from a directory of trajectory files."""
    if topology_file is not None and topology_pattern is not None:
        raise ValueError("Exactly one of pdb_file and pdb_pattern should be provided.")

    # Compile the regex patterns
    traj_pattern_compiled = re.compile(traj_pattern)
    if topology_pattern is not None:
        topology_pattern_compiled = re.compile(topology_pattern)

    # Find all trajectory files recursively
    traj_files = collections.defaultdict(list)
    codes = set()

    for dirpath, _, filenames in os.walk(root):
        rel_dirpath = os.path.relpath(dirpath, root)
        for filename in filenames:
            filepath = os.path.join(rel_dirpath, filename)
            match = traj_pattern_compiled.match(filepath)
            if match:
                code = match.group(1)
                code = os.path.basename(code)
                codes.add(code)
                traj_files[code].append(filepath)

    if len(codes) == 0:
        raise ValueError("No codes found in directory.")

    # Find all topology (.pdb or .sdf) files recursively
    topology_files = {}
    if topology_pattern is not None:
        for dirpath, _, filenames in os.walk(root):
            rel_dirpath = os.path.relpath(dirpath, root)
            for filename in filenames:
                filepath = os.path.join(rel_dirpath, filename)
                match = topology_pattern_compiled.match(filepath)
                if match:
                    code = match.group(1)
                    code = os.path.basename(code)
                    if code in codes:
                        topology_files[code] = filepath
    else:
        for code in codes:
            topology_files[code] = topology_file

    # Filter out codes.
    codes = filter_and_subset_codes(
        codes,
        filter_codes=filter_codes,
        filter_codes_csv=filter_codes_csv,
        filter_codes_csv_header=filter_codes_csv_header,
        max_datasets=max_datasets,
        max_datasets_offset=max_datasets_offset,
    )

    if len(codes) == 0:
        raise ValueError("No codes found after filtering.")

    # Determine dataset class.
    if as_sdf:
        dataset_fn = lambda code: MDtrajSDFDataset(  # noqa: E731
            root,
            traj_files=traj_files[code],
            sdf_file=topology_files[code],
            label=code,
            **dataset_kwargs,
        )
    else:
        if as_iterable:
            dataset_fn = lambda code: MDtrajIterableDataset(  # noqa: E731
                root,
                traj_files=traj_files[code],
                pdb_file=topology_files[code],
                label=code,
                **dataset_kwargs,
            )
        else:
            dataset_fn = lambda code: MDtrajDataset(  # noqa: E731
                root,
                traj_files=traj_files[code],
                pdb_file=topology_files[code],
                label=code,
                **dataset_kwargs,
            )

    # Create datasets.
    datasets = []
    for code in tqdm(codes, desc="Creating datasets"):
        # Skip codes without pdb files.
        if code not in topology_files:
            print(f"Warning: No topology file found for code {code}, skipping.")
            continue

        dataset = dataset_fn(code)
        datasets.append(dataset)

    return datasets


def parse_sdf_datasets_from_directory(
    root: str,
    traj_pattern: str,
    sdf_pattern: str,
    max_datasets: int | None = None,
    max_datasets_offset: int | None = None,
    filter_codes: Sequence[str] | None = None,
    filter_codes_csv: str | None = None,
    filter_codes_csv_header: str | None = None,
    **dataset_kwargs,
) -> list[MDtrajDataset]:
    """Helper function to create MDtrajDataset objects from a directory of trajectory files."""
    # Compile the regex patterns
    traj_pattern_compiled = re.compile(traj_pattern)

    # Find all trajectory files recursively
    traj_files = collections.defaultdict(list)
    codes = set()

    for dirpath, _, filenames in os.walk(root):
        rel_dirpath = os.path.relpath(dirpath, root)
        for filename in filenames:
            filepath = os.path.join(rel_dirpath, filename)
            match = traj_pattern_compiled.match(filepath)
            if match:
                code = match.group(1)
                code = os.path.basename(code)
                codes.add(code)
                traj_files[code].append(filepath)

    if len(codes) == 0:
        raise ValueError("No codes found in directory.")

    # Filter out codes.
    codes = filter_and_subset_codes(
        codes,
        filter_codes=filter_codes,
        filter_codes_csv=filter_codes_csv,
        filter_codes_csv_header=filter_codes_csv_header,
        max_datasets=max_datasets,
        max_datasets_offset=max_datasets_offset,
    )

    if len(codes) == 0:
        raise ValueError("No codes found after filtering.")

    # Create datasets.
    datasets = []
    for code in tqdm(codes, desc="Creating datasets"):
        dataset = MDtrajSDFDataset(
            root,
            sdf_files=traj_files[code],
            label=code,
            **dataset_kwargs,
        )
        datasets.append(dataset)

    return datasets


def filter_and_subset_codes(
    codes: list[str],
    filter_codes: Sequence[str] | None,
    filter_codes_csv: str | None,
    filter_codes_csv_header: str | None,
    max_datasets: int | None,
    max_datasets_offset: int | None,
):
    """Get a list of codes from the dataset."""

    if filter_codes_csv is not None:
        if filter_codes is not None:
            raise ValueError("Only one of filter_codes and filter_codes_csv should be provided.")

        filter_codes = pd.read_csv(filter_codes_csv)[filter_codes_csv_header].tolist()

    if filter_codes is not None:
        filter_codes = set(filter_codes)
        codes = [code for code in codes if code in filter_codes]

    # Sort the codes and offset them, if necessary
    codes = list(sorted(codes))
    if max_datasets_offset is not None:
        codes = codes[max_datasets_offset:]
    if max_datasets is not None:
        codes = codes[:max_datasets]

    return codes


def concatenate_datasets(datasets: Sequence[Sequence[MDtrajDataset]]) -> list[MDtrajDataset]:
    """Concatenate multiple lists of datasets into one list."""
    all_datasets = []
    for datasets_list in datasets:
        all_datasets.extend(datasets_list)
    return all_datasets


def create_dataset_from_pdbs(pdb_entries: dict[str, str], label_prefix: str | None = None) -> Sequence[MDtrajDataset]:
    """Create a dataset from a PDB file."""
    datasets = []
    for label, pdb_file in pdb_entries.items():
        # Note that if pdb_file is an absolute path, the first part of the join will be ignored.
        root = os.path.join(hydra.utils.get_original_cwd(), os.path.dirname(pdb_file))
        pdb_file = os.path.basename(pdb_file)

        if label_prefix is not None:
            label = f"{label_prefix}{label}"

        dataset = MDtrajDataset(
            root=root,
            traj_files=[pdb_file],
            pdb_file=pdb_file,
            label=label,
        )
        datasets.append(dataset)

    return datasets
