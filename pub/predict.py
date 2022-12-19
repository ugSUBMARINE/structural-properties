import os
import sys
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import statsmodels.api as sm

from property_analysis import (
    SB_DATA_DESC,
    HY_DATA_DESC,
    HB_DATA_DESC,
    HB_SELE,
    HY_SELE,
    SB_SELE,
)
from run_rmsf_analysis import temp_cd, activity, p_names
from properties import SaltBridges, HydrophobicClusterOwn

np.set_printoptions(threshold=sys.maxsize)

# ----------------------- PARAMETERS ------------------------------------
# names of proteins for which data is available but are not in original p_names
p_names = np.append(p_names, ["769bc", "N0"])
# name of the saved model in saved_models/
model_name = "af_all"
# data of proteins the model should use for predictions
data_dir = "af_all_out"
# replace None with file_path.json to store output - will be stored in 'results' 
save_results = None
# -----------------------------------------------------------------------


# reading and calculating data for each protein
salt_bridges_data = []
h_bonds_data = []
hydrophobic_cluster_data = []
for i in p_names:
    hb = SaltBridges(os.path.join(f"{data_dir}/{i}/h_bonds/{i}.csv"), -1, HB_DATA_DESC)
    h_bonds_data.append(list(hb.stats())[:-1])

    sb = SaltBridges(
        os.path.join(f"{data_dir}/{i}/saltbridges/{i}.csv"), -1, SB_DATA_DESC
    )
    salt_bridges_data.append(list(sb.stats()[:-1]))

    hy = HydrophobicClusterOwn(
        os.path.join(f"{data_dir}/{i}/hydrophobic_cluster/{i}.csv"),
        HY_DATA_DESC,
    )
    hydrophobic_cluster_data.append(list(hy.stats())[:-1])
print("data calculation done")

salt_bridges_data = np.asarray(salt_bridges_data)
h_bonds_data = np.asarray(h_bonds_data)
hydrophobic_cluster_data = np.asarray(hydrophobic_cluster_data)
# create DataFrames
sb_df = pd.DataFrame(salt_bridges_data, index=p_names, columns=SB_DATA_DESC).round(2)
hb_df = pd.DataFrame(h_bonds_data, index=p_names, columns=HB_DATA_DESC).round(2)
hy_df = pd.DataFrame(
    hydrophobic_cluster_data, index=p_names, columns=HY_DATA_DESC
).round(2)

# attributes used
hb_vals = HB_SELE
hy_vals = HY_SELE
sb_vals = SB_SELE
# make one big DataFrame
master_frame = pd.concat(
    [hb_df[hb_vals], hy_df[hy_vals], sb_df[sb_vals]],
    axis=1,
)

# reading the saved parameters for the model and loading the model
param_file = open(f"saved_models/{model_name}_setting.txt", "r")
model_param = param_file.readline().strip().split(",")
param_file.close()
data = np.asarray(master_frame.loc[:, model_param])
model = sm.load(f"saved_models/{model_name}.pickle")

# using the model to predict the data of interest
predictions = model.predict(np.column_stack((np.ones(len(data)), data)))
prediction_order = np.argsort(predictions)
# print(np.mean(np.abs(temp_cd - predictions)))
# print(stats.pearsonr(temp_cd, predictions))
for i, j in zip(p_names[prediction_order], predictions[prediction_order]):
    print(f"{i:<6}: {j:0.1f}")
print(" < ".join(p_names[prediction_order]))

if save_results is not None:
    if not os.path.isdir("results"):
        os.mkdir("results")
    res = dict(zip(p_names, predictions))
    with open(os.path.join("results", save_results), "w") as res_file:
        json.dump(res, res_file)
