import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm


from property_analysis import (
    SB_DATA_DESC,
    HY_DATA_DESC,
    HB_DATA_DESC,
    create_best_model,
)
from run_rmsf_analysis import temp_cd, activity, p_names
from properties import SaltBridges, HydrophobicClusterOwn

np.set_printoptions(threshold=sys.maxsize)
plt.style.use("default")
plt.rcParams.update({"font.size": 23})
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)


def single_stats(
    desc: list[str],
    property_data: np.ndarray[tuple[int, int], np.dtype[int | float]],
    cv: list[int | float] | np.ndarray[tuple[int], np.dtype[int | float]],
    num_params: int = 3,
) -> np.ndarray[tuple[int], np.dtype[str]]:
    """calculate and print correlation between single properties and experimental data
    :parameter
        - desc:
          name of the properties that will be looked attributes
        - property_data:
          calculated values for each property
        - cv:
          experimental data to compare against
        - num_params:
          how many parameters should be return
    :return
        - descriptions
          *_DATA_DESC with the highest PearsonR
    """
    descriptions, pr, pp = [], [], []
    for i in range(len(desc)):
        i_data = property_data[:, i]
        if len(np.unique(i_data)) > 1:
            ipr, ipp = stats.pearsonr(cv, i_data)
            print(f"--- {desc[i]} ---")
            print(f"PearsonR: {ipr:>8.4f}\np: {ipp:>15.4f}\nstd: {i_data.std():13.4f}")
            print(f"Max: {np.max(i_data):13.4f}\nMin: {np.min(i_data):13.4f}")
            descriptions.append(desc[i])
            pr.append(ipr)
            pp.append(ipp)
    order = np.lexsort((pp, -np.abs(pr)))
    chosen = np.asarray(descriptions)[order][:num_params]
    print(f"Chosen vals: {' / '.join(chosen.tolist())}")
    return chosen


def fit_and_plot(
    data_path,
    model_ind: int | str | None = None,
    save_model: str | None = None,
    num_replicas: int = 1,
    target: np.ndarray[tuple[int], np.dtype[int | float]] = temp_cd,
    show_plots: bool = False,
    save_plots: bool = False,
    add_names: list[str] | None = None,
    add_temp: list[int] | None = None,
    hb_param_num: int = 3,
    hy_param_num: int = 3,
    sb_param_num: int = 3,
    ow_hb_vals: np.ndarray[tuple[int], np.dtype[str]] | None = None,
    ow_hy_vals: np.ndarray[tuple[int], np.dtype[str]] | None = None,
    ow_sb_vals: np.ndarray[tuple[int], np.dtype[str]] | None = None,
    p_names_in: np.ndarray[tuple[int], np.dtype[str]] | None = None,
) -> None:
    """fit models, find the best parameter combination and plot scatter for multi data
    per protein
    :parameter
        - data_path:
          path where the data to all proteins is stored
        - model_ind:
          index of which model to test and save or
          e.g. "!4" to specify the number of parameters or None to use the best model
        - save_model:
          how the saved model should be named - None to not save
        - num_replicas:
          number of structures per protein
        - target:
          data to fit the model to
        - show_plots:
          only possible for num_replicas > 1
        - save_plots:
          only possible for num_replicas > 1
        - add_names:
          to add a protein name to the used names e.g. "769bc"
        - add_temp:
          data of the added protein e.g. 75.3
        - hb_param_num:
          number of H-Bonds attributes to be tested
        - hy_param_num:
          number of Hydrophobic Cluster attributes to be tested
        - sb_param_num:
          number of Salt Bridges attributes to be tested
        - ow_hb_vals:
          if chosen H-Bonds attributes in *_vals should be overwritten
        - ow_hy_vals:
          if chosen Hydrophobic Cluster attributes in *_vals should be overwritten
        - ow_sb_vals:
          if chosen Salt Bridges attributes in *_vals should be overwritten
        - p_names_in:
          Name of the proteins and their respective files like 769bc for 769bc.pdb
    :return
        - None
    """
    if add_names is None:
        add_names = []
    if add_temp is None:
        add_temp = []
    if p_names_in is None:
        global p_names
    else:
        p_names = p_names_in

    target = np.append(target, add_temp)
    print(" < ".join(target[np.argsort(target)].astype(str)))
    p_names = np.append(p_names, add_names)
    # name of the folders where the data is stored
    data_folders = ["saltbridges", "h_bonds", "hydrophobic_cluster"]
    data_pos = dict(zip(data_folders, np.arange(len(data_folders))))
    data = []
    for i in data_folders:
        data.append([])

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
            if len(inter_data) == 1:
                data[data_pos[a]].append(inter_data[0])
            else:
                multi_data[data_pos[a]].append(inter_data)
                data[data_pos[a]].append(np.median(inter_data, axis=0))

    salt_bridges_data = np.asarray(data[data_pos["saltbridges"]])
    h_bonds_data = np.asarray(data[data_pos["h_bonds"]])
    hydrophobic_cluster_data = np.asarray(data[data_pos["hydrophobic_cluster"]])

    print("Hydrogen Bonds")
    hb_vals = single_stats(HB_DATA_DESC, h_bonds_data, target, hb_param_num)
    print("\nHydrophobic Cluster")
    hy_vals = single_stats(HY_DATA_DESC, hydrophobic_cluster_data, target, hy_param_num)
    print("\nSalt Bridges")
    sb_vals = single_stats(SB_DATA_DESC, salt_bridges_data, target, sb_param_num)

    # to be able to overwrite the *_vals
    if ow_hb_vals is not None:
        hb_vals = ow_hb_vals
    if ow_hy_vals is not None:
        hy_vals = ow_hy_vals
    if ow_sb_vals is not None:
        sb_vals = ow_sb_vals

    # create DataFrames
    sb_df = pd.DataFrame(salt_bridges_data, index=p_names, columns=SB_DATA_DESC).round(
        2
    )
    hb_df = pd.DataFrame(h_bonds_data, index=p_names, columns=HB_DATA_DESC).round(2)
    hy_df = pd.DataFrame(
        hydrophobic_cluster_data, index=p_names, columns=HY_DATA_DESC
    ).round(2)

    create_best_model(
        hb_df[hb_vals],
        hy_df[hy_vals],
        sb_df[sb_vals],
        hb_vals,
        hy_vals,
        sb_vals,
        target,
        p_names,
        save_plots,
        show_plots,
        save_model=save_model,
        chose_model_ind=model_ind,
    )

    if num_replicas > 1 and (show_plots or save_plots):
        # can have higher indices than shown by single_stats because it shows
        # only values which have more than one unique value
        hb_ind = [int(np.argwhere(HB_DATA_DESC == i)) for i in hb_vals]
        hy_ind = [int(np.argwhere(HY_DATA_DESC == i)) for i in hy_vals]
        sb_ind = [int(np.argwhere(SB_DATA_DESC == i)) for i in sb_vals]

        fig, ax = plt.subplots(
            len(data_folders),
            max([hb_param_num, hy_param_num, sb_param_num]),
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
                        [data[i][p][a]] * 2,
                        color="black",
                        marker="x",
                        linewidth=2,
                    )
                    ax[i, ca].set_title(dn)
                    ax[i, ca].set_ylabel(ddes[a])
                    if i == len(data_folders) - 1:
                        ax[i, ca].set_xticks(
                            np.arange(len(p_names)), p_names, rotation=45
                        )
                    else:
                        ax[i, ca].tick_params(bottom=False, labelbottom=False)
        fig.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
        if save_plots:
            fig.savefig("att_scatter.png")
        if show_plots:
            plt.show()


if __name__ == "__main__":
    pass
    for i in [5, 6, 7]:
        fit_and_plot("esm_d_true_out/", model_ind=f"!{i}", save_model=f"esm_d_true_{i}")
