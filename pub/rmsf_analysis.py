import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from regressor_analysis import (
    SB_DATA_DESC,
    HY_DATA_DESC,
    HB_DATA_DESC,
)
from properties import SaltBridges, HydrophobicClusterOwn

np.set_printoptions(threshold=sys.maxsize)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 25})


def data_coord_extraction(
    target_pdb_file: str,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[str]],
    np.ndarray[tuple[int, int], np.dtype[float]],
]:
    """Reads the content of a given pdb file
    parameter:
        target_pdb_file: pdb file with data of protein of interest
    return:
        res_data: 2d list [[Atom type, Residue 3letter, ChainID, ResidueID],...]
        res_coords: 2d list of corresponding coordinates to the new_data entries
    """
    # list of all data of the entries like
    # [[Atom type, Residue 3letter, ChainID, ResidueID],...]
    res_data = []
    # list of all coordinates of the entries like [[x1, y1, z1],...]
    res_coords = []
    # reading the pdb file
    file = open(target_pdb_file, "r")
    for line in file:
        if "ATOM  " in line[:6]:
            line = line.strip()
            res_data += [
                [
                    line[12:16].replace(" ", "").strip(),
                    line[17:20].replace(" ", "").strip(),
                    line[21].replace(" ", "").strip(),
                    line[22:26].replace(" ", "").strip(),
                ]
            ]
            res_coords += [
                [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]
            ]
    file.close()

    res_data = np.asarray(res_data)
    res_coords = np.asarray(res_coords, dtype=float)
    return res_data, res_coords


def clustalw_alignment_parser(
    file_path: str, num_seq: int, head_lines: int = 3, lines_to_skip: int = 2
) -> np.ndarray[tuple[int], np.dtype[str]]:
    """extracts each aligned sequence of a sequence alignment and joins it to
        return each aligned sequence as a string
    parameter:
        file_path: path to the sequence alignment as str
        num_seq: number of sequences in the alignment as int
        head_lines: number of lines until the first sequence line starts as int
        lines_to_skip: number of lines between the blocks of sequence lines as int
    :return
        final_seq: numpy array of strings like ["A-VL", "AI-L", "AIV-"]"""
    alignment = open(file_path)
    seqs = []
    for i in range(num_seq):
        seqs += [[]]
    alignment_list = list(alignment)
    alignment.close()
    # indices of which lines to use from the alignment file to reconstruct the sequences
    lines_to_use = []
    lines_encountered = 0
    lines_skipped = 0
    for i in range(head_lines, len(alignment_list)):
        if lines_encountered < num_seq:
            lines_to_use += [i]
            lines_encountered += 1
        else:
            lines_skipped += 1
        if lines_skipped == lines_to_skip:
            lines_encountered = 0
            lines_skipped = 0
    pre_sequences = np.asarray(alignment_list)[lines_to_use]

    line_count = 0
    for i in pre_sequences:
        seqs[line_count] += [i.strip().split("\t")[0].split(" ")[-1]]
        line_count += 1
        if line_count == num_seq:
            line_count = 0

    final_seq = []
    for i in seqs:
        final_seq += ["".join(i)]
    return np.asarray(final_seq)


def read_dat(file_path: str) -> pd.DataFrame:
    """reads .dat file from SimulationInteractionsDiagram analysis of
    Schrödinger Desmond
    parameter:
        file_path: path to .dat file as str
    :return
        full_data_df: .dat file as a pandas dataframe"""
    rmsf_file = open(file_path)
    full_data = []
    colum_names = []
    for ci, i in enumerate(rmsf_file):
        line = i.strip().split(" ")
        j_data = []
        for j in line:
            if len(j) > 0:
                j_data += [j.strip()]
        if ci == 0:
            colum_names += j_data
        else:
            full_data += [j_data]
    rmsf_file.close()
    full_data_df = pd.DataFrame(full_data, columns=colum_names[1:])
    return full_data_df


def return_seq_pdb(
    file_path: str, chains: str | None = None
) -> np.ndarray[tuple[int, int], np.dtype[str]]:
    """get protein sequence as present in the pdb file in file_path
    input:
        file_path: path to pdb file as str\n
        chains: list of chains that should be used eg ['A', 'B']
        - if None all chains will be used
    :return
        pdb_seq_sorted: numpy array that contains
        [[Residue 3letter, ChainID, ResidueID],...] of either all chains if chains=None
        or of the specified chains in chains
    """
    d, _ = data_coord_extraction(file_path)
    pdb_seq = np.unique(d[:, 1:], axis=0)
    pdb_seq_sorted = pdb_seq[np.lexsort((pdb_seq[:, 2].astype(int), pdb_seq[:, 1]))]
    if chains is not None:
        return pdb_seq_sorted[np.isin(pdb_seq_sorted[:, 1], np.asarray(chains))]
    else:
        return pdb_seq_sorted


def align_scores(sequences: list[str], seq_values: list[list[int]]):
    """inserts np.Nan in list of seq_values when '-' in sequence
    input:
        sequences: list of sequences from an alignment as strings
        eg ["A-VL", "AI-L", "AIV-"]
        seq_values: list of lists with corresponding scores to the sequences
        eg [[3, 3, 3], [2, 2, 2], [1, 1, 1]]
    :return
        scores: numpy array(s) with added np.Nan where a '-' is present in the sequences
    """
    # sequences = ["A-VL", "AI-L", "AIV-"]
    # seq_values = [[3, 3, 3], [2, 2, 2], [1, 1, 1]]
    index_counts = np.zeros(len(sequences)).astype(int)
    scores = []
    for i in range(len(sequences)):
        scores += [[]]
    for i in range(len(sequences[0])):
        curr_pair = []
        for j in sequences:
            curr_pair += [j[i]]
        gap = np.asarray(curr_pair) != "-"
        for k in range(len(gap)):
            if gap[k]:
                scores[k] += [seq_values[k][index_counts[k]]]
            else:
                scores[k] += [np.nan]
        index_counts[gap] += 1
    return np.asarray(scores)


def map_bfactor(
    pdb_filepath: str,
    result_filepath: str,
    b_factor: list[int | float] | np.ndarray[tuple[int], np.dtype[int | float]],
    chain_oi: list[str],
    rmsf_norm: bool = True,
) -> None:
    """map the normaliued b-factor or rmsf into the pdb b-factor column
    :parameter
        - pdb_filepath:
          /PATH/TO/PDB_FILE.pdb
        - result_filepath:
          /PATH/TO/RESULT_FILE.pdb
        - b_factor:
          data for each atom or for each residue
        - chain_oi:
          list of chains the values should be mapped on eg ['A', 'B']
        - rmsf_norm:
          set to True if each value in b_factor corresponds to one residue or False if
          each value corresponds to one atom
    :return
        - None
    """
    a = open(pdb_filepath, "r")
    data = a.readlines()
    a.close()

    print(
        "minimum: {:0.4f}\nmaximum:{:0.4f}".format(np.min(b_factor), np.max(b_factor))
    )

    c = -1
    prev_res = None
    n = open(
        result_filepath,
        "w+",
    )
    for line in data:
        if "ATOM  " in line[:6]:
            name = line[17:20].replace(" ", "").strip()
            chain = line[21].replace(" ", "").strip()
            num = line[22:26].replace(" ", "").strip()
            if chain in chain_oi:
                cur_res = "".join([name, chain, num])
                if rmsf_norm:
                    if prev_res != cur_res:
                        c += 1
                        prev_res = cur_res
                else:
                    c += 1
                b_write = f"{b_factor[c]:0.2f}"
                line_list = list(line)
                for i in range(1, 7):
                    try:
                        line_list[65 - (i - 1)] = b_write[-i]
                    except IndexError:
                        line_list[65 - (i - 1)] = " "
                n.write("".join(line_list))
        else:
            n.write(line)
    n.close()


def plot_multi_files(
    data_path,
    hb_vals: list[str],
    hy_vals: list[str],
    sb_vals: list[str],
    num_replicas: int = 10,
    show_plots: bool = False,
    save_plots: bool = False,
):
    """plot scatter plot for multiple files per protein
    :parameter
        - data_path:
          path where the data to all proteins is stored
        - hb_vals, hy_vals, sb_vals:
          attributes to plot like in *_DATA_DESC
        - num_replicas:
          number of structures per protein
        - show_plots:
          only possible for num_replicas > 1
        - save_plots:
          only possible for num_replicas > 1
    :return
        - None
    """
    p_names = np.asarray(
        ["4alb", "N2", "N4", "N5", "N31", "N55", "N80", "N122", "N124", "N134"]
    )
    # name of the folders where the data is stored
    data_folders = ["saltbridges", "h_bonds", "hydrophobic_cluster"]
    data_pos = dict(zip(data_folders, np.arange(len(data_folders))))

    multi_data = []
    for i in data_folders:
        multi_data.append([])

    # for each protein
    for i in p_names:
        # for each data attribute
        for a in data_folders:
            # csv dir path
            c_path = os.path.join(data_path, i, a)
            # all csv files
            files = os.listdir(c_path)
            # calculate data
            inter_data = []
            for c in files:
                ac_path = os.path.join(c_path, c)
                if a == "saltbridges":
                    sb = SaltBridges(ac_path, -1, SB_DATA_DESC)
                    inter_data.append(list(sb.stats())[:-1])
                elif a == "h_bonds":
                    hb = SaltBridges(ac_path, -1, HB_DATA_DESC)
                    inter_data.append(list(hb.stats()[:-1]))
                else:
                    hy = HydrophobicClusterOwn(ac_path, HY_DATA_DESC)
                    inter_data.append(list(hy.stats())[:-1])
            # store data depending on one or multiple files per protein
            multi_data[data_pos[a]].append(inter_data)

    # can have higher indices than shown by single_stats because it shows
    # only values which have more than one unique value
    hb_ind = [int(np.argwhere(HB_DATA_DESC == i)) for i in hb_vals]
    hy_ind = [int(np.argwhere(HY_DATA_DESC == i)) for i in hy_vals]
    sb_ind = [int(np.argwhere(SB_DATA_DESC == i)) for i in sb_vals]

    fig, ax = plt.subplots(
        len(data_folders),
        # max([hb_param_num, hy_param_num, sb_param_num]),
        3,
        figsize=(32, 18),
    )
    # p_names = [i.replace("4alb", "BsPAD") for i in p_names]
    for i in range(len(data_folders)):
        if data_folders[i] == "h_bonds":
            att_inds = hb_ind
            dn = "Hydrogen Bonds"
            ddes = HB_DATA_DESC
        elif data_folders[i] == "hydrophobic_cluster":
            att_inds = hy_ind
            dn = "Hydrophobic Cluster"
            ddes = HY_DATA_DESC
        elif data_folders[i] == "saltbridges":
            att_inds = sb_ind
            dn = "Salt Bridges"
            ddes = SB_DATA_DESC
        else:
            raise KeyError("Invalid data folder encountered")
        for p in range(len(p_names)):
            for ca, a in enumerate(att_inds):
                ax[i, ca].scatter(
                    [p] * num_replicas,
                    np.asarray(multi_data[i][p])[:, a],
                    label=p_names[p],
                )
                ax[i, ca].plot(
                    [p - 0.2, p + 0.2],
                    [np.median(np.asarray(multi_data[i][p])[:, a])] * 2,
                    color="black",
                    marker="x",
                    linewidth=2,
                )
                ax[i, ca].set_title(dn)
                ax[i, ca].set_ylabel(ddes[a])
                if i == len(data_folders) - 1:
                    ax[i, ca].set_xticks(np.arange(len(p_names)), p_names, rotation=45)
                else:
                    ax[i, ca].tick_params(bottom=False, labelbottom=False)
    fig.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
    if save_plots:
        fig.savefig("att_scatter.png")
    if show_plots:
        plt.show()


if __name__ == "__main__":
    pass

    plot_multi_files(
        "md_sim_structures/",
        np.asarray(["MEAN NWS HB", "SUM NWS HB", "SUM BPN HB"]),
        np.asarray(["MEAN CC", "MAX CA", "MAX CC"]),
        np.asarray(["MEAN NWS SB", "SUM NWS SB", "SUM IA SB"]),
        show_plots=True,
        save_plots="scatter.png"
    )
