import os
import sys
import numpy as np

# add parent dir to path to be able to import find_h_bonds etc.
sys.path.append((os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from h_bond_test import find_h_bonds
from salt_bridges import find_saltbridges
from hydrophobic_cluster import hydr_cluster
from surface import surface_amino_acids

from regressor_analysis import read_data


def check_dir(path: str) -> None:
    """check if directory exists and create if it doesn't
    :parameter
        - path:
          path to directory of interest
    :retur
        - None
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def calculate(
    out_dir: str,
    struct_dir: str,
    p_names_in: np.ndarray[tuple[int], np.dtype[str]] = None,
):
    """calculate properties for all given structures
    :parameter
        - out_dir:
          output directory name
        - struct_dir:
          path to the directory containing all structures
        - p_names_in:
          names of the proteins (and theirfore their file names)

    :return
        - func1return
          description
    """
    # create dir to store results
    check_dir(out_dir)
    for i in p_names_in:
        # to be compatible with structures files
        h_path = os.path.join(struct_dir, i + "_H.pdb")
        if os.path.isfile(h_path):
            file_path = h_path
        else:
            file_path = os.path.join(struct_dir, i + ".pdb")
        print(file_path)

        # calculate H-bonds, salt bridges, hydrophobic cluster and store their files
        find_h_bonds(
            file_path,
            create_file=f"{out_dir}/{i}_hb",
            silent=True,
        )
        find_saltbridges(
            file_path,
            create_file=f"{out_dir}/{i}_sb",
            silent=True,
        )
        hydr_cluster(
            file_path,
            create_file=f"{out_dir}/{i}_hy",
            silent=True,
        )
        surface_amino_acids(
            file_path,
            create_file=f"{out_dir}/{i}_sf",
            silent=True,
        )


if __name__ == "__main__":
    pass
    p_names, _ = read_data()
    """
    p_names = p_names[:-2]
    for i in range(10):
        calculate(f"./md_attr/md_{i}_out", f"./md_attr/md_{i}/", p_names)
    """
    calculate("test_out_", "af_all_ii/", p_names)

    # calculate("structures_out_", "structures/", p_names)
