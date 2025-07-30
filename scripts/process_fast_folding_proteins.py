import argparse
import logging
import os
import random
import shutil
import sys

import tqdm

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("process_fast_folding_proteins")


SPLITS = {
    "train": 0.85,
    "val": 0.5,
    "test": 0.1,
}


def find_pdb_file(root_dir: str) -> str:
    """
    Recursively find the first .pdb file in the directory tree starting from root_dir

    Args:
        root_dir (str): The root directory to start the search from

    Returns:
        str: Full path to the first .pdb file found, or None if no .pdb file is found
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        py_logger.info(f"Searching in: {dirpath}")

        for filename in filenames:
            if filename.endswith("filtered.pdb") and not filename.startswith("."):
                pdb_file_path = os.path.join(dirpath, filename)
                return pdb_file_path

    return None


def find_xtc_files(root_dir: str) -> dict[str, str]:
    """
    Recursively find all .xtc files and their containing folders starting from root_dir

    Args:
        root_dir (str): The root directory to start the search from

    Returns:
        Dict[str, str]: A dictionary where keys are "containing_folder_filename" and values are full paths to .xtc files
    """
    xtc_files = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        py_logger.info(f"Searching in: {dirpath}")

        # Find all files with .xtc extension.
        for filename in filenames:
            if not filename.endswith(".xtc") or filename.startswith("."):
                continue

            # Get full file path and containing folder.
            file_path = os.path.join(dirpath, filename)
            containing_folder = os.path.basename(dirpath)
            key = f"{containing_folder}_{filename}"

            assert key not in xtc_files, f"Duplicate key found: {key}"
            xtc_files[key] = file_path

    return xtc_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create splits of .xtc files based on containing folder")
    parser.add_argument(
        "--inputdir", help="Directory where original trajectories were downloaded", type=str, required=True
    )
    parser.add_argument("--outputdir", "-o", help="Output directory to save splits", type=str, required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.inputdir):
        py_logger.info(f"Error: Directory '{args.inputdir}' does not exist", file=sys.stderr)
        sys.exit(1)

    pdb_file = find_pdb_file(args.inputdir)
    if not pdb_file:
        raise ValueError("No filtered.pdb PDB file found in the directory.")
    py_logger.info(f"Found PDB file: {pdb_file}")

    # Copy the PDB file to the output directory.
    os.makedirs(args.outputdir, exist_ok=True)
    shutil.copy(pdb_file, os.path.join(args.outputdir, "filtered.pdb"))
    py_logger.info(f"Copied PDB file to {args.outputdir}")

    files = find_xtc_files(args.inputdir)
    if not files:
        raise ValueError("No .xtc files found in the directory.")
    py_logger.info(f"Found {len(files)} .xtc files in the directory.")

    # Randomly shuffle the files and folders.
    random.seed(42)
    files = list(sorted(files.items(), key=lambda x: x[0]))
    random.shuffle(files)

    # Now, create the splits based on the folders.
    for split, split_ratio in SPLITS.items():
        split_dir = os.path.join(args.outputdir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Find the number of files to include in this split.
        num_files = int(len(files) * split_ratio)
        split_files = files[:num_files]

        # Save the split files as a text file.
        with open(os.path.join(split_dir, "files.txt"), "w") as f:
            for key, input_file in split_files:
                f.write(f"{input_file}\n")

        # Copy the files to the split directory.
        for key, input_file in tqdm.tqdm(split_files, desc=f"Copying files for {split} split"):
            output_file = os.path.join(split_dir, f"{key}.xtc")
            shutil.copy(input_file, output_file)

        py_logger.info(f"Saved {len(split_files)} files for {split} split in {split_dir}")
