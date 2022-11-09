import os
import sys
import argparse

import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def data_coord_extraction(
    target_pdb_file: str,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[str]],
    np.ndarray[tuple[int, int], np.dtype[float]],
]:
    """extracts the coordinates and the residue data from a pdb file
    :parameter
         - target_pdb_file:
           path to pdb file for protein of interest
    :returns
         - new_data: 2D ndarray
           contains information about all residues like [[Atom type,
           Residue 3letter, ChainID, ResidueID],...]
         - new_coords:
           contains coordinates of corresponding residues to the new_data
           entries
    """
    # list of all data of the entries like [[Atom type, Residue 3letter,
    # ChainID, ResidueID],...]
    res_data = []
    # list of all coordinates of the entries like [[x1, y1, z1],...]
    res_coords = []
    # reading the pdb file
    file = open(target_pdb_file, "r")
    for line in file:
        if "ATOM  " in line[:6]:
            line = line.strip()
            res_data.append(
                [
                    line[12:16].replace(" ", "").strip(),
                    line[17:20].replace(" ", "").strip(),
                    line[21].replace(" ", "").strip(),
                    line[22:26].replace(" ", "").strip(),
                ]
            )
            res_coords.append(
                [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]
            )
    file.close()

    res_data = np.asarray(res_data)
    res_coords = np.asarray(res_coords, dtype=float)
    return res_data, res_coords


def dist_calc(
    arr1: np.ndarray[tuple[int, int], np.dtype[int | float]],
    arr2: np.ndarray[tuple[int, int], np.dtype[int | float]],
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """calculates distance between arr1 and arr2 and returns a 2D array with
    all distances of all arr1 points against all arr2 points
     :parameter
         - arr1, arr2:
           2D arrays of 1D lists with 3D coordinates eg [[x1, y1, z1],...]
     :return
         - dist:
           len(arr1) x len(arr2) distance matrix between arr1 and arr2"""
    # get only the x,y,z coordinates from the input arrays and reshape them,
    # so they can be subtracted from each other
    arr1_coords_rs = arr1.reshape(arr1.shape[0], 1, arr1.shape[1])
    arr2_coord_rs = arr2.reshape(1, arr2.shape[0], arr1.shape[1])
    # calculating the distance between each point and returning a 2D array
    # with all distances
    dist = np.sqrt(((arr1_coords_rs - arr2_coord_rs) ** 2).sum(axis=2))
    return dist


def c_cluster(pair_num: list[list[str]]) -> list[list[str]]:
    """clusters pairs of interactions that feature on common residue
    :parameter
        - pair_num:
          pairs of salt bridge forming residues as their residue number
    :return
        - pair_num:
          all residues that are in the same cluster in one list

    """
    pair_num = pair_num.tolist()
    # which residues are forming salt bridges and how often they are in contact
    # with another residue
    multi_ap, mac = np.unique(pair_num, return_counts=True)
    # dict describing the number of contacts for each residue
    app_dict = dict(zip(multi_ap, mac))

    """
    while the number of clusters changes due to merging
       for all existing clusters (starts with pairs)
            if the residues in the current cluster appear only in one cluster
                put this cluster in the clusters list
            else
                check for each cluster if one of the residues of the current
                cluster that's looked at is in another cluster
                if that's the case
                    put the current cluster and the one where one of the
                    residues are in in the inter_cluster
                make the residues in inter_cluster unique
                if this cluster is not in the clusters list put it in there
            change the clusters that are looked at to the new created clusters
            list and restart if the number of clusters is not the same as in
            the previous round
    """
    prev_len = 0
    while True:
        clusters = []
        for i in pair_num:
            if np.sum(list(map(app_dict.get, i))) == len(i):
                clusters.append(i)
            else:
                inter_cluster = []
                for k in pair_num:
                    for j in i:
                        if j in k:
                            inter_cluster += i
                            inter_cluster += k
                potential_cluster = np.unique(inter_cluster).tolist()
                if potential_cluster not in clusters:
                    clusters.append(potential_cluster)
        pair_num = clusters
        c_len = len(clusters)
        if c_len == prev_len:
            break
        else:
            prev_len = c_len
    return pair_num


def join_res_data(
    part0: np.ndarray[tuple[int, int], np.dtype[str]],
    part1: np.ndarray[tuple[int, int], np.dtype[str]],
) -> np.ndarray[tuple[int, int], np.dtype[str]]:
    joined_data = []
    for i in range(len(part0)):
        joined_data.append(["-".join(part0[i]), "-".join(part1[i])])
    return np.asarray(joined_data)


def create_output(
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
    for i in pair_num:
        inter_pairs = []
        for k in i:
            inter_pairs += pair_num_ori[np.where(pair_num_ori == k)[0]].tolist()
        cpc.append(np.unique(inter_pairs, axis=0).shape[0])
        r = []
        for f in np.unique(inter_pairs):
            f_split = f.split("-")
            r.append(f_split[-1])
    # output
    if create_file is not None:
        data_file = open(f"{create_file}.csv", "w+")
        data_file.write("InteractingResidues,ContactsPerCluster\n")
    for i in range(len(pair_num)):
        if not silent:
            print(" - ".join(pair_num[i]), cpc[i])
        if create_file is not None:
            data_file.write(",".join([" - ".join(pair_num[i]), str(cpc[i])]) + "\n")
    if create_file is not None:
        data_file.close()


def find_saltbridges(
    file_path: str,
    max_dist: int | float = 3.5,
    sele_chain: str = None,
    create_file: str = None,
    silent: bool = False,
):
    """searches for saltbridges in a given structure with the option to limit the chains
    :parameter
        - file_path:
          path to the pdb file
        - sele_chain:
          to select a specific chain use e.g. 'A' in which the salt bridges should be
          calculated
        - create_file:
          how the output file will be named - gets split if '_' present
        - silent:
          whether to print output in terminal or not
    :return
        - None
    """
    data, coords = data_coord_extraction(file_path)
    # ARG NH1, NH2
    # LYS NZ
    # HIS ND1 NE2
    # ASP OD1 OD2
    # GLU OE1 OE2
    aa = ["ARG", "LYS", "HIS", "ASP", "GLU"]
    atom = ["NH1", "NH2", "NZ", "ND1", "NE2", "OD1", "OD2", "OE1", "OE2"]
    charge = {"ARG": 1, "LYS": 1, "HIS": 1, "ASP": -1, "GLU": -1}
    tests = []
    for i in aa:
        tests.append((data[:, 1] == i).tolist())
    for i in atom:
        tests.append((data[:, 0] == i).tolist())
    tests = np.asarray(tests)
    # which data entry contains the right amino acid and the right atom type
    test_conf = np.sum(tests, axis=0) == 2
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
    # all atoms in salt bridge distance
    dists = dists < max_dist
    # to only get interactions once
    dists = np.triu(dists, 1)
    # indices of which sele_data are in interacting distance
    pairs = np.where(dists)
    # charge of the interacting residues
    charge_map = np.asarray(list(map(charge.get, sele_data[:, 1])))
    # only allow interacting pairs to be valid if their charge is opposite
    valid_charge_bool = charge_map[pairs[0]] + charge_map[pairs[1]] == 0
    # ResidueIDs of the valid interaction pairs
    valid_pairs = np.column_stack(
        (sele_data[:, [1, 2, 3]][pairs[0]], sele_data[:, [1, 2, 3]][pairs[1]])
    )[valid_charge_bool]
    valid_pairs = np.unique(valid_pairs, axis=0)

    # pairs of salt bridge forming residues as their residue number as strings
    pair_num = join_res_data(valid_pairs[:, 3:], valid_pairs[:, :3])
    pair_num_ori = pair_num

    pair_num = c_cluster(pair_num)

    create_output(pair_num, pair_num_ori, create_file=create_file, silent=silent)
    return pair_num_ori


def arg_dict() -> dict:
    """argparser for salt bridges search
    :parameter
        - None:
    :return
        - d
          dictionary specifying all parameters for find_saltbridges
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--file_path", type=str, required=True, help="path to pdb file"
    )
    parser.add_argument(
        "-d",
        "--max_dist",
        type=float,
        required=False,
        help="max distance for valid interactions",
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
        help="ChainID if if the salt bridges should "
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
    find_saltbridges(**arg_dict())
