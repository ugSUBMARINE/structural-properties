import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm


from property_analysis import (
    SB_DATA_DESC,
    HY_DATA_DESC,
    HB_DATA_DESC,
    create_best_model,
    HB_SELE,
    HY_SELE,
    SB_SELE,
)
from run_rmsf_analysis import temp_cd, activity, p_names
from properties import SaltBridges, HydrophobicClusterOwn

np.set_printoptions(threshold=sys.maxsize)
plt.rcParams.update({"font.size": 23})
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)


def single_stats(
    desc: list[str],
    property_data: np.ndarray[tuple[int, int], np.dtype[int | float]],
    cv: list[int | float] | np.ndarray[tuple[int], np.dtype[int | float]],
) -> None:
    """calculate and print correlation between single properties and experimental data
    :parameter
        - desc:
          name of the properties that will be looked attributes
        - property_data:
          calculated values for each property
        - cv:
          experimental data to compare against
    :return
        - None
    """
    for i in range(len(desc)):
        i_data = property_data[:, i]
        if len(np.unique(i_data)) > 1:
            ipr, ipp = stats.pearsonr(cv, i_data)
            print(desc[i])
            print(f"PearsonR: {ipr:>8.4f}\np: {ipp:>15.4f}")


# ----------------------- PARAMETERS ------------------------------------
show_plots = False
save_plots = False
# -----------------------------------------------------------------------

# name of the folders where the data is stored
data_folders = ["saltbridges", "h_bonds", "hydrophobic_cluster"]
# how proteins are originally ordered (by p_names)
p_name_order = dict(zip(p_names, np.arange(len(p_names))))
# in which order the protein files are read and stored
order = []
salt_bridges_data = []
h_bonds_data = []
hydrophobic_cluster_data = []

fig, ax = plt.subplots(3, 3, figsize=(32, 18))
fig_kde, ax_kde = plt.subplots(3, 3, figsize=(32, 18))

# iterate over all directories where the data for each protein and frame is stored
for subdir, dirs, files in os.walk(os.path.expanduser("md_sim_structures/")):
    # which data we are currently reading
    data_kind = os.path.split(subdir)[-1]
    if data_kind in data_folders:
        # get the salt bridge frame
        if data_kind == data_folders[0]:
            cur_protein = os.path.split(os.path.split(subdir)[0])[-1]
            order.append(cur_protein)
            sb_data = []
            # calculation of salt bridge data for each frame
            for f in files:
                sb = SaltBridges(os.path.join(subdir, f), -1, SB_DATA_DESC)
                sb_data.append(list(sb.stats())[:-1])
            salt_bridges_data.append(np.median(sb_data, axis=0))
            sb_data = np.asarray(sb_data, dtype=float)
            hb_num = len(HB_SELE)
            for ci, i in enumerate(SB_SELE):
                hb_num = ci
                # scatter plot
                ax[2, hb_num].scatter(
                    [p_name_order[cur_protein]] * 10, sb_data[:, i], label=cur_protein
                )
                ax[2, hb_num].plot(
                    [p_name_order[cur_protein] - 0.2, p_name_order[cur_protein] + 0.2],
                    [np.median(sb_data[:, i])] * 2,
                    linewidth=2,
                    marker="x",
                    color="black",
                )
                ax[2, hb_num].set_title("SaltBridges")
                ax[2, hb_num].set_ylabel(SB_DATA_DESC[i])
                ax[2, ci].set_xticks(np.arange(len(p_names)), p_names, rotation=45)

                # kernel density estimation
                kde = sm.nonparametric.KDEUnivariate(sb_data[:, i])
                kde.fit()
                ax_kde[2, hb_num].fill(kde.support, kde.density, alpha=0.3)
                ax_kde[2, hb_num].plot(kde.support, kde.density, label=cur_protein)
                ax_kde[2, hb_num].set_ylabel("Density")
                ax_kde[2, hb_num].set_xlabel(SB_DATA_DESC[i])

        # get hydrogen bond frame
        if data_kind == data_folders[1]:
            cur_protein = os.path.split(os.path.split(subdir)[0])[-1]
            hb_data = []
            # calculation of hydrogen bond data for each frame
            for f in files:
                hb = SaltBridges(os.path.join(subdir, f), -1, HB_DATA_DESC)
                hb_data.append(list(hb.stats()[:-1]))
            h_bonds_data.append(np.median(hb_data, axis=0))
            hb_data = np.asarray(hb_data)
            for ci, i in enumerate(HB_SELE):
                # scatter plot
                ax[0, ci].scatter(
                    [p_name_order[cur_protein]] * 10, hb_data[:, i], label=cur_protein
                )
                ax[0, ci].plot(
                    [p_name_order[cur_protein] - 0.2, p_name_order[cur_protein] + 0.2],
                    [np.median(hb_data[:, i])] * 2,
                    linewidth=2,
                    marker="x",
                    color="black",
                )
                ax[0, ci].set_title("Hydrogen Bonds")
                ax[0, ci].set_ylabel(HB_DATA_DESC[i])
                ax[0, ci].tick_params(bottom=False, labelbottom=False)

                # kernel density estimation
                kde = sm.nonparametric.KDEUnivariate(hb_data[:, i])
                kde.fit()
                ax_kde[0, ci].fill(kde.support, kde.density, alpha=0.3)
                ax_kde[0, ci].plot(kde.support, kde.density, label=cur_protein)
                ax_kde[0, ci].set_ylabel("Density")
                ax_kde[0, ci].set_xlabel(HB_DATA_DESC[i])

        # get hydrophobic cluster frame
        if data_kind == data_folders[2]:
            cur_protein = os.path.split(os.path.split(subdir)[0])[-1]
            hy_data = []
            # calculation of hydrophobic cluster data for each frame
            for f in files:
                hy = HydrophobicClusterOwn(os.path.join(subdir, f), HY_DATA_DESC)
                hy_data.append(list(hy.stats())[:-1])
            hydrophobic_cluster_data.append(np.median(hy_data, axis=0))
            hy_data = np.asarray(hy_data)
            for ci, i in enumerate(HY_SELE):
                # scatter plot
                ax[1, ci].scatter(
                    [p_name_order[cur_protein]] * 10, hy_data[:, i], label=cur_protein
                )
                ax[1, ci].plot(
                    [p_name_order[cur_protein] - 0.2, p_name_order[cur_protein] + 0.2],
                    [np.median(hy_data[:, i])] * 2,
                    linewidth=2,
                    marker="x",
                    color="black",
                )
                ax[1, ci].set_title("HydrophobicCluster")
                ax[1, ci].set_ylabel(HY_DATA_DESC[i])
                ax[1, ci].tick_params(bottom=False, labelbottom=False)

                # kernel density estimation
                kde = sm.nonparametric.KDEUnivariate(hy_data[:, i])
                kde.fit()
                ax_kde[1, ci].fill(kde.support, kde.density, alpha=0.3)
                ax_kde[1, ci].plot(kde.support, kde.density, label=cur_protein)
                ax_kde[1, ci].set_ylabel("Density")
                ax_kde[1, ci].set_xlabel(HY_DATA_DESC[i])

salt_bridges_data = np.asarray(salt_bridges_data)
h_bonds_data = np.asarray(h_bonds_data)
hydrophobic_cluster_data = np.asarray(hydrophobic_cluster_data)

fig.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
if save_plots:
    fig.savefig("scatter_plot.png")

order = np.asarray(order)
# list of where proteins should be according to original order
order_order = list(map(p_name_order.get, order))
# indices to order proteins in the original order
ori_order_ind = np.argsort(order_order)

# create nice legend
leg_lines, leg_labels = ax_kde[0, 0].get_legend_handles_labels()
fig_kde.legend(
    [leg_lines[ind] for ind in ori_order_ind],
    [leg_labels[ind] for ind in ori_order_ind],
    loc="lower center",
    ncol=10,
)
fig_kde.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
if save_plots:
    fig_kde.savefig("kde_plot.png")

# order data according to original order
salt_bridges_data = salt_bridges_data[ori_order_ind, :]
h_bonds_data = h_bonds_data[ori_order_ind, :]
hydrophobic_cluster_data = hydrophobic_cluster_data[ori_order_ind, :]

# create DataFrames
sb_df = pd.DataFrame(salt_bridges_data, index=p_names, columns=SB_DATA_DESC).round(2)
hb_df = pd.DataFrame(h_bonds_data, index=p_names, columns=HB_DATA_DESC).round(2)
hy_df = pd.DataFrame(
    hydrophobic_cluster_data, index=p_names, columns=HY_DATA_DESC
).round(2)

print("Hydrogen Bonds")
single_stats(HB_DATA_DESC, h_bonds_data, temp_cd)
print("\nHydrophobic Cluster")
single_stats(HY_DATA_DESC, hydrophobic_cluster_data, temp_cd)
print("\nSalt Bridges")
single_stats(SB_DATA_DESC, salt_bridges_data, temp_cd)

create_best_model(hb_df, hy_df, sb_df, temp_cd, save_plots, show_plots)
