import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

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


if __name__ == "__main__":
    pass
