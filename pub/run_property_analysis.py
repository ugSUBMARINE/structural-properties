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


# ----------------------- PARAMETERS ------------------------------------
# only possible for num_replicas > 1
show_plots = False
save_plots = False
# number of structures per protein
num_replicas = 10
# index of which model to test and save or e.g. "!4" to specify the number of parameters
model_ind = None  # "!6" # 1
# how the saved model should be named - None to not save
save_model = None  # "esm_single"  # None
# which data the model should use for fitting
data_path = "esm_double_out"
# to add a protein name to the used names e.g. "769bc" - empty [] to not add
add_names = []
# data of the added protein e.g. 75.3 - empty [] to not add
add_temp = []
# data to fit the model to (ndarray[tuple[int], np.dtype[int|float]])
target = temp_cd
hb_param_num = 3
hy_param_num = 3
sb_param_num = 3
# -----------------------------------------------------------------------

target = np.append(target, add_temp)
print(" < ".join(target[np.argsort(target)].astype(str)))
p_names = np.append(p_names, add_names)
# name of the folders where the data is stored
data_folders = ["saltbridges", "h_bonds", "hydrophobic_cluster"]
# how proteins are originally ordered (by p_names)
p_name_order = dict(zip(p_names, np.arange(len(p_names))))


# name of the folders where the data is stored
data_folders = ["saltbridges", "h_bonds", "hydrophobic_cluster"]
data_pos = dict(zip(data_folders, np.arange(len(data_folders))))
data = []
for i in data_folders:
    data.append([])

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
        if len(inter_data) == 1:
            data[data_pos[a]].append(inter_data[0])
        else:
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

# create DataFrames
sb_df = pd.DataFrame(salt_bridges_data, index=p_names, columns=SB_DATA_DESC).round(2)
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
