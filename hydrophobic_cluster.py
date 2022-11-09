import sys
import os
import argparse

import numpy as np

from salt_bridges import data_coord_extraction, dist_calc, c_cluster, join_res_data

np.set_printoptions(threshold=sys.maxsize)

# surface accessible side chain area
sasa = {
    "ALA": 75,
    "CYS": 115,
    "ASP": 130,
    "GLU": 161,
    "PHE": 209,
    "GLY": 0,
    "HIS": 180,
    "ILE": 172,
    "LYS": 205,
    "LEU": 172,
    "MET": 184,
    "ASN": 142,
    "PRO": 134,
    "GLN": 173,
    "ARG": 236,
    "SER": 95,
    "THR": 130,
    "VAL": 143,
    "TRP": 254,
    "TYR": 222,
}


def create_output_hy(
    pair_num: list[list[str]],
    pair_num_ori: list[list[str]],
    create_file: str = None,
    silent: bool = False,
) -> None:
    """prints output in the terminal and optionally creates a output file
    :parameter
        - pair_num:
          clustered residues in one list
        - pair_num_ori:
          the original pairs of interacting residues
        - create_file:
          how the output file will be named
        - silent:
          whether to print output in terminal or not
    :return
        - None
    """
    # list with cluster sizes
    cluster_sizes = [len(i) for i in pair_num]
    # contacts per cluster_sizes
    cpc = []
    # surface area per cluster
    spc = []
    for i in pair_num:
        inter_pairs = []
        for k in i:
            inter_pairs += pair_num_ori[np.where(pair_num_ori == k)[0]].tolist()
        cpc.append(np.unique(inter_pairs, axis=0).shape[0])
        r = []
        surface_area = 0
        for f in np.unique(inter_pairs):
            f_split = f.split("-")
            surface_area += int(sasa[f_split[0]])
            r.append(f_split[-1])
        spc.append(surface_area)
    if create_file is not None:
        data_file = open("{}.csv".format(create_file), "w+")
        data_file.write(
            "InteractingResidues,ContactsPerCluster,SurfaceAreaPerCluster\n"
        )
    for i in range(len(pair_num)):
        if not silent:
            print(" - ".join(pair_num[i]), cpc[i], spc[i])
        if create_file is not None:
            data_file.write(
                ",".join([" - ".join(pair_num[i]), str(cpc[i]), str(spc[i])]) + "\n"
            )
    if create_file is not None:
        data_file.close()


def hydr_cluster(
    file_path: str,
    sele_chain: str = None,
    create_file: str = None,
    silent: bool = False,
) -> None:
    """calculates hydrophobic cluster with the option to select a chain
    :parameter
        - file_path:
          path to the pdb file
        - sele_chain:
          to select a specific chain use e.g. 'A' in which the salt bridges
          should be calculated
        - create_file:
          how the output file will be named - gets split if '_' present
        - silent:
          whether to print output in terminal or not
    :return
        - None
    """
    data, coords = data_coord_extraction(file_path)
    aa = ["ILE", "LEU", "VAL"]
    atom = ["N", "H", "CA", "HA", "C", "O"]
    tests = []
    for i in aa:
        tests.append((data[:, 1] == i).tolist())
    for i in atom:
        tests.append((data[:, 0] != i).tolist())
    heavy_atom = []
    for i in data[:, 0]:
        heavy_atom.append(not i.startswith("H"))
    tests.append(heavy_atom)
    tests = np.asarray(tests)
    # which data entry contains the right amino acid and the right atom type
    test_conf = np.sum(tests, axis=0) == 8
    if sele_chain is None:
        chain_test = np.ones(data[test_conf].shape[0]).astype(bool)
    else:
        # to get data entries from selected chain(s)
        chain_test = data[test_conf][:, 2] == sele_chain
    # ResidueIDs for interacting residues that are able to form bridges
    sele_data = data[test_conf][chain_test]
    # their coordinates
    sele_coords = coords[test_conf][chain_test]
    # distance matrix between all the potential residues
    dists = dist_calc(sele_coords, sele_coords)
    # all atoms in hydrophobic interaction distance
    dists = dists < 6.56
    # to only get interactions once
    dists = np.triu(dists, 1)
    pair_ind0, pair_ind1 = np.where(dists)
    excl_same = np.any(
        sele_data[:, [1, 2, 3]][pair_ind0] != sele_data[:, [1, 2, 3]][pair_ind1], axis=1
    )
    # pairs of hydrophobic interactions with Amino Acid, Chain and Number
    valid_pairs = np.column_stack(
        (
            sele_data[:, [1, 2, 3]][pair_ind0][excl_same],
            sele_data[:, [1, 2, 3]][pair_ind1][excl_same],
        )
    )
    # to only have one entry per residues interaction and not of all their
    # atoms
    valid_pairs = np.unique(valid_pairs, axis=0)

    # pairs of cluster forming residues as their residue number as strings
    pair_num = join_res_data(valid_pairs[:, 3:], valid_pairs[:, :3])
    pair_num_ori = pair_num

    pair_num = c_cluster(pair_num)

    create_output_hy(pair_num, pair_num_ori, create_file=create_file, silent=silent)
    return pair_num_ori


def arg_dict() -> dict:
    """argparser for hydrophobic cluster search
    :parameter
        - None:
    :return
        - d
          dictionary specifying all parameters for hydr_cluster
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--file_path", type=str, required=True, help="path to pdb file"
    )
    parser.add_argument(
        "-c", "--create_file", type=str, required=False, default=None, help="file name"
    )
    parser.add_argument(
        "-s",
        "--sele_chain",
        type=str,
        required=False,
        default=None,
        help="ChainID if the hydrophobic cluster should "
        "only calculated for one specific chain",
    )
    args = parser.parse_args()
    d = {
        "file_path": args.file_path,
        "create_file": args.create_file,
        "sele_chain": args.sele_chain,
    }
    return d


if __name__ == "__main__":
    hydr_cluster(**arg_dict())
