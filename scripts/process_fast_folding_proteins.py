import argparse
import logging
import os
import random
import shutil
import sys
import tqdm
from typing import List, Tuple, Dict

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("process_fast_folding_proteins")


SPLITS = {
    "train": 0.85,
    "val": 0.5,
    "test": 0.1,
}


def find_xtc_files(root_dir: str) -> Dict[str, str]:
    """
    Recursively find all .xtc files and their containing folders starting from root_dir

    Args:
        root_dir (str): The root directory to start the search from

    Returns:
        Dict[str, str]: A dictionary where keys are "containing_folder_filename" and values are full paths to .xtc files
    """
    xtc_files = {}

    # Walk through directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        py_logger.info(f"Searching in: {dirpath}")

        # Find all files with .xtc extension
        for filename in filenames:
            if not filename.endswith(".xtc") or filename.startswith("."):
                continue

            # Get full file path and containing folder
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

    # Check if directory exists
    if not os.path.isdir(args.inputdir):
        py_logger.info(f"Error: Directory '{args.inputdir}' does not exist", file=sys.stderr)
        sys.exit(1)

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
            # py_logger.info(f"Saved {input_file} as {output_file}")
        
        py_logger.info(f"Saved {len(split_files)} files for {split} split in {split_dir}")
