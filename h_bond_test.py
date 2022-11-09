import os
import argparse
from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt
from salt_bridges import (
    data_coord_extraction,
    dist_calc,
    c_cluster,
    join_res_data,
    create_output,
)


def twod_angle(
    acc: np.ndarray[tuple[int, int], int | float],
    h: np.ndarray[tuple[int, int], int | float],
    d: np.ndarray[tuple[int, int], int | float],
) -> np.ndarray[tuple[int], np.dtype[float]]:
    """calculate angle between three points in 3D space
    :parameter
        - acc:
          acceptor coordinates
        - h:
          hydrogen coordinates
        - d:
          donor coordinates
    :return
        - theta:
          angle in radians between the acc - h - d
    """
    # get vectors
    p_arr0 = acc - h
    p_arr1 = d - h
    # all against all dot product
    dotp = np.sum(p_arr0 * p_arr1, axis=1)
    # vector normalization
    pa0_norm = np.linalg.norm(p_arr0, axis=1)
    pa1_norm = np.linalg.norm(p_arr1, axis=1)
    # cosine of the angle
    cosine_theta = np.clip(dotp / (pa0_norm * pa1_norm), -1.0, 1.0)
    # angle in radians
    theta = np.arccos(cosine_theta)
    return theta


def find_h_bonds(
    file_path: str,
    create_file: str = None,
    d_max: int | float = 2.5,
    ang_min: int | float = 120.0,
    d_xh: int | float = 1.1,
    donor_atoms: str = "N O F",
    sele_chain: str = None,
    silent: bool = False,
) -> None:
    """calculate hydrogen bonds between all residues that are capable of h-bonding
    :parameter
        - file_path:
          path to the pdb file
        - create_file:
          if not None should be the name of the protein - gets split by '_'
        - d_max:
          max distance in \u212B between Donor H and Acceptor
        - ang_min:
          min angle between Donor - H - Acceptor
        - d_xh:
          max distance in \u212B between Donor and H
        - donor_atoms:
          single character for donor atom separated by a space per atom
        - sele_chain:
          ChainID if if the H- bonds should only calculated for one specific chain
        - silent:
          whether to print output in terminal or not
    :return
        - None
    """
    # get residue data and coordinates from the pdb file
    data, coords = data_coord_extraction(file_path)

    # use either whole data or only of the selected chain of sele_chain
    if sele_chain is not None:
        chain_test = data[:, 2] == sele_chain
    else:
        chain_test = np.ones(len(data)).astype(bool)
    data = data[chain_test]
    coords = coords[chain_test]

    # atom of each entry
    atom_data = np.array([i[0] for i in data[:, 0]])

    # bool which entry in data is a donor/ acceptor atom
    d_bool = np.isin(atom_data, donor_atoms.split())
    # bools which entry in data is a H atom
    h_bool = atom_data == "H"

    # indices for all data
    inds = np.arange(coords.shape[0])
    # indices for all H atoms in data
    h_inds = inds[h_bool]
    # indices for all donor / acceptor atoms in data
    ha_inds = inds[d_bool]

    # distances between all H atoms and all none H atoms
    dist_h_assign = dist_calc(coords[ha_inds], coords[h_inds])

    # ha_inds with H atom near a donor_atoms
    ha_assign = []
    # inds for which h_ind is at a donor_atoms
    h_donor = []
    # for each h_inds find the nearest donor_atoms ha_inds that is in the same
    # residues - this is the one it is bonded to
    dist_sort = np.argsort(dist_h_assign, axis=0)
    for i in range(h_inds.shape[0]):
        h_res = data[h_inds[i]][1:]
        for j in dist_sort[:, i]:
            j_test = data[ha_inds[j]]
            if np.all(j_test[1:] == h_res):
                if dist_h_assign[j, i] < d_xh:
                    h_donor.append(i)
                    ha_assign.append(ha_inds[j])
                break
    ha_assign = np.asarray(ha_assign)
    h_donor = np.asarray(h_donor)

    # inds for ha_inds that are donor_atoms not bound to a H
    not_assigned = np.invert(np.isin(ha_inds, ha_assign))
    full_unassign = np.logical_and(not_assigned, d_bool[ha_inds])
    ha_unassign = np.where(full_unassign)[0]

    # inds for data/coords
    # H atom of donor
    hdh_true_inds = h_inds[h_donor]
    # donor_atoms of donor
    hdha_true_inds = ha_assign
    # acceptor atoms
    ha_true_inds = ha_inds[ha_unassign]

    # all H bound to a donor_atoms against all acceptor atoms
    # ha_part - which ha_true_inds makes the distance cutoff
    # hd_part - which hdh_true_inds/hdh_true_inds makes the distance cutoff
    ha_part, hd_part = np.where(dist_h_assign[np.ix_(ha_unassign, h_donor)] <= d_max)

    # ture inds selected for making the distance cutoff
    ha_donor_inds = hdha_true_inds[hd_part]
    hh_donor_inds = hdh_true_inds[hd_part]
    h_acceptor_inds = ha_true_inds[ha_part]

    # angle calculation between donor_atoms - H - acceptor atoms that are in the
    # right distance
    angle_calc = twod_angle(
        coords[h_acceptor_inds], coords[hh_donor_inds], coords[ha_donor_inds]
    )

    # which donor-acceptor pair fulfills the angel criteria
    angle_bool = angle_calc > np.radians(ang_min)

    # creating output and output files of valid donor-acceptor pairs
    part0 = data[ha_donor_inds][angle_bool][:, [1, 2, 3]]
    part1 = data[h_acceptor_inds][angle_bool][:, [1, 2, 3]]
    pair_num_ori = join_res_data(part0, part1)
    pair_num = c_cluster(pair_num_ori)

    create_output(pair_num, pair_num_ori, create_file, silent=silent)
    return pair_num_ori


def arg_dict() -> dict:
    """argparser for H-bond search
    :parameter
        - None:
    :return
        - d
          dictionary specifying all parameters for find_h_bonds
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
        "-dm",
        "--d_max",
        type=float,
        required=False,
        default=2.5,
        help="max distance in \u212B between Donor H and Acceptor",
    )
    parser.add_argument(
        "-a",
        "--ang_min",
        type=float,
        required=False,
        default=120.0,
        help="min angle between Donor - H - Acceptor",
    )
    parser.add_argument(
        "-dx",
        "--d_xh",
        type=float,
        required=False,
        default=1.1,
        help="max distance in \u212B between Donor and H",
    )
    parser.add_argument(
        "-da",
        "--donor_atoms",
        type=str,
        required=False,
        default="N O F",
        help="single character for donor atom separated by a space per atom",
    )
    parser.add_argument(
        "-s",
        "--sele_chain",
        type=str,
        required=False,
        default=None,
        help="ChainID if if the H- bonds should only calculated for one specific chain",
    )
    args = parser.parse_args()
    d = {
        "file_path": args.file_path,
        "create_file": args.create_file,
        "d_max": args.d_max,
        "ang_min": args.ang_min,
        "d_xh": args.d_xh,
        "donor_atoms": args.donor_atoms,
        "sele_chain": args.sele_chain,
    }
    return d


if __name__ == "__main__":
    find_h_bonds(**arg_dict())
