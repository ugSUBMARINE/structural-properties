import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from rmsf_analysis import (
    clustalw_alignment_parser,
    read_dat,
    return_seq_pdb,
    align_scores,
    map_bfactor,
)

np.set_printoptions(threshold=sys.maxsize)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 25})

one_three = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

three_one = {v: k for k, v in one_three.items()}

# needs to be the same order as in the alignment file
p_names = np.asarray(
    ["4alb", "N2", "N4", "N5", "N31", "N55", "N80", "N122", "N124", "N134"]
)
p_colors = [
    (174 / 256, 174 / 256, 174 / 256),
    (249 / 256, 231 / 256, 127 / 256),
    (63 / 256, 90 / 256, 150 / 256),
    (202 / 256, 223 / 256, 185 / 256),
    (242 / 256, 205 / 256, 177 / 256),
    (166 / 256, 195 / 256, 227 / 256),
    (66 / 256, 88 / 256, 45 / 256),
    (177 / 256, 36 / 256, 25 / 256),
    (105 / 256, 52 / 256, 155 / 256),
    (2 / 256, 1 / 256, 2 / 256),
]

chains = np.asarray(
    [
        ["A", "B"],
        ["A", "B"],
        ["A", "B"],
        ["A", "B"],
        ["A", "B"],
        ["A", "D"],
        ["A", "B"],
        ["A", "B"],
        ["A", "B"],
        ["C", "D"],
    ]
)

temp_cd = np.asarray([54.5, 56.1, 55.2, 46.4, 78.1, 57.8, 65.4, 60.9, 55.3, 72.7])
activity = np.asarray([20, 2.5, 5, 3, 4, 12, 4, 6, 2, 1.5])

if __name__ == "__main__":
    # ----------------------- PARAMETERS ------------------------------------
    # which column of the dat file should be used
    col_oi = "CA"
    # how many residues should be converted to np.nan from 0:cut_from_start
    # and from - cut_from_end:
    cut_from_start = 15
    cut_from_end = 25
    # index of p_names which protein should be used as baseline in the second plot
    # and for the rmsf value search
    baseline_ind = 0
    # reference chain from the baseline_ind when searching for rmsf values of
    # specific residues
    chain_oi = "A"
    # residues for which the rmsf value should be searched
    # - indexed like in the pdb file
    res_oi = None  #  [68, 66, 98, 41, 64, 31, 11, 13]
    # where a region of residues in the alignment start in the pdb 0 indexed
    start_roi = None  # 115
    end_roi = None  # 140
    # number of replicas
    num_replica = 10
    # whether to print the alignment or not
    show_alignment = False
    # whether plots should be shown
    show_plots = False
    # whether to save the plots
    save_plots = False
    # file name of the alignment file
    algn_file = "all_ancestor_alignment_per_chain.clustal_num"
    # name of the directory containing the protein structures
    structure_path = "structures"
    # name of the directory containing Schrödinger's .dat files
    data_path = "rmsf"
    # base name of the .dat files
    file_base = "P_RMSF"
    # -----------------------------------------------------------------------
    # mean RMSF values per residue for all proteins from the 10 replicas per protein
    mean_res_vals = []
    # mean_res_vals split per chain
    res_vals_per_chain = []
    # lengths of the chains as present in the pdb file
    pdb_chain_len = []
    # sequences from the pdb file as str 'AVLI'
    real_seq = []
    # pdb file content of selected chain [[THR A 68], ...]
    pdb_file_content = []
    for cp, p in enumerate(p_names):
        # mean over all 10 replicas
        inter_vals = []
        for i in range(1, num_replica + 1):
            inter_vals += [
                list(read_dat(f"{data_path}/{p}/{file_base}{i}.dat")[col_oi])
            ]
        # mean of all replicas of protein p
        mean_inter_vals = np.mean(np.asarray(inter_vals).astype(float), axis=0)
        mean_res_vals.append(mean_inter_vals)
        # get sequences from the pdb files of specified chains
        pdb_cont = return_seq_pdb(
            os.path.join(structure_path, p + ".pdb"),
            chains[cp],
        )
        pdb_file_content.append(pdb_cont)
        # length of the first chain
        prev_len = None
        for cc, c in enumerate(np.unique(pdb_cont[:, 1])):
            # change the 3-letter code from the pdb file to one-letter code
            chain_seq = "".join(
                list(map(three_one.get, pdb_cont[:, 0][pdb_cont[:, 1] == c]))
            )
            real_seq.append(chain_seq)
            # length of one chain
            cl = len(chain_seq)
            pdb_chain_len.append(cl)
            if cc == 0:
                prev_len = cl
                res_vals_per_chain.append(mean_inter_vals[:cl].tolist())
            else:
                res_vals_per_chain.append(mean_inter_vals[prev_len:].tolist())

    # each proteins chain lengths as own list
    pdb_chain_len = np.split(np.asarray(pdb_chain_len), len(p_names))

    # sequences from alignment with '-'
    aligned_seq = clustalw_alignment_parser(algn_file, 20)
    if show_alignment:
        for ci, i in enumerate(aligned_seq):
            print(f"{p_names[ci // 2]:>4} {i}")
        print()

    # added np.nan to the list of RMSF values where "-" is present in the alignment
    nan_filled = align_scores(aligned_seq, res_vals_per_chain)

    chain_mean_rmsf = []
    for ci, i in enumerate(np.split(np.arange(len(aligned_seq)), len(p_names))):
        # nan filled values of both aligned chains
        chain_mean_val = np.nanmean(nan_filled[i], axis=0)
        # change ends to np.nan
        chain_mean_val[:cut_from_start] = np.nan
        chain_mean_val[-cut_from_end:] = np.nan
        chain_mean_rmsf += [chain_mean_val.tolist()]

    chain_mean_rmsf = np.asarray(chain_mean_rmsf)

    # plot to compare the RMSF values of each protein against all other proteins
    fig, ax = plt.subplots(2, 5, sharex="col", sharey="row", figsize=(32, 18))
    c = 0
    for i in range(len(p_names)):
        if i % 2 == 0:
            row = 0
            if i > 0:
                c += 1
        else:
            row = 1
        for j in range(len(p_names)):
            if j != i:
                ax[row, c].plot(chain_mean_rmsf[j], color="silver", alpha=0.2)
        ax[row, c].plot(
            chain_mean_rmsf[i], label=p_names[i], color=p_colors[i], linewidth=2.5
        )
        ax[row, c].legend()
        if row == 1:
            ax[row, c].set_xlabel("Residue Index")
        if c == 0:
            ax[row, c].set_ylabel("Mean RMSF")
    fig.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
    if save_plots:
        plt.savefig("rmsf_plot.png")
    if show_plots:
        plt.show()

    # RMSF differences
    print(f"Differences in the RMSF to the selected baseline {p_names[baseline_ind]}")
    # which arrays are not the baseline
    base_pool = np.arange(len(p_names)) != baseline_ind
    for ci, i in enumerate(chain_mean_rmsf[base_pool]):
        # difference to RMSF of proteins that is used as baseline
        diff_to_base = i - chain_mean_rmsf[baseline_ind]
        i_name = p_names[base_pool][ci]
        n_sum = np.nansum(diff_to_base)

        print(
            "{:>5} RMSF sum: {:>8} -|- RMSF difference to baseline {:>5}%".format(
                i_name,
                np.round(n_sum, 3),
                np.round((np.sum(diff_to_base > 0.0) / len(i)) * 100, 0),
            )
        )
    print()

    # average RMSF per residue
    cut_vals = []
    variances = []
    stds = []
    for ci, i in enumerate(res_vals_per_chain):
        i_work = np.asarray(i[cut_from_start:-cut_from_end])
        variances.append(np.var(i_work))
        stds.append(i_work.std())
        cut_vals.append(np.sum(i_work) / len(i_work))
    cut_vals = np.asarray(cut_vals)
    variances = np.asarray(variances)
    stds = np.asarray(stds)
    # get the values per protein in one row
    mean_protein_rmsf = np.mean(np.asarray(np.split(cut_vals, len(p_names))), axis=1)
    mean_vars = np.mean(np.asarray(np.split(variances, len(p_names))), axis=1)
    mean_stds = np.mean(np.asarray(np.split(stds, len(p_names))), axis=1)
    protein_rmsf_sort = np.argsort(mean_protein_rmsf)

    # getting the sequence/ pdb positions of residue regions in the alignment
    if end_roi is not None and start_roi is not None:
        print()
        end_roi = end_roi + 1
        chain_count = 0
        for i in range(len(p_names)):
            for j in range(chains.shape[1]):
                insertions = np.asarray(list(aligned_seq[chain_count])) == "-"
                print(
                    f"{p_names[i]:>4} Chain {chains[i][j]} starts at residue position "
                    f"{start_roi - np.sum(insertions[:start_roi]):>3} and ends at "
                    f"{end_roi - np.sum(insertions[:end_roi]) - 1:>3} (0 indexed)"
                )
                chain_count += 1
        print()

    # RMSF values for specific residues of all proteins
    if res_oi is not None:
        # pdb file content of the baseline chain of interest
        current_cont = np.asarray(pdb_file_content[baseline_ind])
        # 2d ndarray with the data or with ['-', '-', '-'] if '-' in seq alignment
        ref_pdb = []
        index_count = 0
        for i in aligned_seq[0]:
            # fill the ref_pdb either with the data or with ['-', '-', '-']
            # if '-' in seq alignment
            if i != "-":
                ref_pdb += [current_cont[current_cont[:, 1] == chain_oi][index_count]]
                index_count += 1
            else:
                ref_pdb += [np.asarray(["-", "-", "-"])]
        ref_pdb = np.asarray(ref_pdb)
        res_means = []
        print("selected residues:")
        for r in res_oi:
            # where in the reference in the alignment the residue r of interest
            # is located
            mask = ref_pdb[:, 2] == str(r)
            print(" ".join(ref_pdb[mask][0].astype(str).tolist()))
            inter_rvals = []
            # getting the values for each sequence (chain) in the alignment
            for i in nan_filled:
                inter_rvals += [i[mask][0]]
            res_means += [
                np.mean(
                    np.split(np.asarray(inter_rvals), len(p_names)), axis=1
                ).tolist()
            ]
        res_means = np.asarray(res_means)

        res_rmsf_sum = np.sum(res_means, axis=0)
        res_rmsf_sort = np.argsort(res_rmsf_sum)

        print("\nCorrelation between temp_cd and res_rmsf_sum")
        pea_res_temp = pearsonr(temp_cd, res_rmsf_sum)
        print(f"PearsonR: {pea_res_temp[0]:>8.4f}\np: {pea_res_temp[1]:>15.4f}")
        print("Correlation between activity and res_rmsf_sum")
        pea_res_act = pearsonr(activity, res_rmsf_sum)
        print(f"PearsonR: {pea_res_act[0]:>8.4f}\np: {pea_res_act[1]:>15.4f}\n")

    print("Correlation between temp_cd and mean_protein_rmsf")
    pea_prot_temp = pearsonr(temp_cd, mean_protein_rmsf)
    print(f"PearsonR: {pea_prot_temp[0]:>8.4f}\np: {pea_prot_temp[1]:>15.4f}")
    print("Correlation between activity and mean_protein_rmsf")
    pea_prot_act = pearsonr(activity, mean_protein_rmsf)
    print(f"PearsonR: {pea_prot_act[0]:>8.4f}\np: {pea_prot_act[1]:>15.4f}\n")

    print("Correlation with temp_cd and variances")
    pea_var_temp = pearsonr(temp_cd, mean_vars)
    print(f"PearsonR: {pea_var_temp[0]:>8.4f}\np: {pea_var_temp[1]:>15.4f}")
    print("Correlation with activity and variances")
    pea_var_act = pearsonr(activity, mean_vars)
    print(f"PearsonR: {pea_var_act[0]:>8.4f}\np: {pea_var_act[1]:>15.4f}\n")

    print("Correlation with temp_cd and stds")
    pea_std_temp = pearsonr(temp_cd, mean_stds)
    print(f"PearsonR: {pea_std_temp[0]:>8.4f}\np: {pea_std_temp[1]:>15.4f}")
    print("Correlation with activity and stds")
    pea_std_act = pearsonr(activity, mean_stds)
    print(f"PearsonR: {pea_std_act[0]:>8.4f}\np: {pea_std_act[1]:>15.4f}\n")
