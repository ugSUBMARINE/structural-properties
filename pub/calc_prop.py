import os
import sys
import numpy as np

# add parent dir to path to be able to import find_h_bonds etc.
sys.path.append((os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from h_bond_test import find_h_bonds
from salt_bridges import find_saltbridges
from hydrophobic_cluster import hydr_cluster

from run_rmsf_analysis import p_names

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


# ----------------------- PARAMETERS ------------------------------------
# output directory name in pub
parent_name = "single_struct_out"
# structure path in pub
struct_pub = "structures"
# to add a protein name to the used names e.g. "769bc" - empty [] to not add
add_names = ["N0", "769bc"]
# -----------------------------------------------------------------------
p_names = np.append(p_names, add_names)

# create dir to store results
check_dir(parent_name)
for i in p_names:
    # base dir for storage of data
    i_base_path = os.path.join(parent_name, i)
    check_dir(i_base_path)

    # to be compatible with structures files
    h_path = os.path.join(struct_pub, i + "_H.pdb")
    if os.path.isfile(h_path):
        file_path = h_path
    else:
        file_path = os.path.join(struct_pub, i + ".pdb")
    print(file_path)

    # calculate H-bonds, salt bridges, hydrophobic cluster and store their files
    hb_path = os.path.join(i_base_path, "h_bonds")
    check_dir(hb_path)
    find_h_bonds(
        file_path,
        create_file=f"{hb_path}/{i}",
        silent=True,
    )

    sb_path = os.path.join(i_base_path, "saltbridges")
    check_dir(sb_path)
    find_saltbridges(
        file_path,
        create_file=f"{sb_path}/{i}",
        silent=True,
    )

    hy_path = os.path.join(i_base_path, "hydrophobic_cluster")
    check_dir(hy_path)
    hydr_cluster(
        file_path,
        create_file=f"{hy_path}/{i}",
        silent=True,
    )
