import os
import sys
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import statsmodels.api as sm
import joblib

from property_analysis import (
    SB_DATA_DESC,
    HY_DATA_DESC,
    HB_DATA_DESC,
)
from run_rmsf_analysis import temp_cd, activity, p_names
from properties import SaltBridges, HydrophobicClusterOwn
from run_property_analysis import get_data

np.set_printoptions(threshold=sys.maxsize)


def predict(
    model_name: str,
    data_dir: str,
    p_names: np.ndarray[tuple[int], np.dtype[str]],
    add: list | None = None,
    save_results: bool = False,
    silent: bool = False,
    loading_method: str = "sm"
) -> None:
    """predict using a saved model
    :parameter
        - model_name:
          name of the saved model in saved_models/
        - data_dir:
          data of proteins the model should use for predictions
        - p_names:
          name of the proteins
        - add:
          names of proteins for which data is available but are not in original p_names
        - save_results:
          True to save results under model_name.json in 'results'
        - silent:
          True to not print any output
        - loading_method:
          *  'sm' for statsmodels saved models
          *  'sk' for models saved with joblib and sklearn
    :return
        - predictions
          predicted values for all p_names
    """
    if add is not None:
        p_names = np.append(p_names, add)
    # reading and calculating data for each protein
    h_bonds_data, hydrophobic_cluster_data, salt_bridges_data = get_data(
        data_dir, p_names_in=p_names
    )
    # create DataFrames
    sb_df = pd.DataFrame(salt_bridges_data, index=p_names, columns=SB_DATA_DESC).round(
        2
    )
    hb_df = pd.DataFrame(h_bonds_data, index=p_names, columns=HB_DATA_DESC).round(2)
    hy_df = pd.DataFrame(
        hydrophobic_cluster_data, index=p_names, columns=HY_DATA_DESC
    ).round(2)

    # make one big DataFrame
    master_frame = pd.concat([hb_df, hy_df, sb_df], axis=1)

    # reading the saved parameters for the model and loading the model
    param_file = open(f"saved_models/{model_name}_setting.txt", "r")
    model_param = param_file.readline().strip().split(",")
    param_file.close()

    data = np.asarray(master_frame.loc[:, model_param])
    model_file_path = f"saved_models/{model_name}.pickle"
    if loading_method == "sm":
        model = sm.load(model_file_path)
        predictions = model.predict(np.column_stack((np.ones(len(data)), data)))
    elif loading_method == "sk":
        model = joblib.load(model_file_path)
        predictions = model.predict(data)

    # using the model to predict the data of interest
    prediction_order = np.argsort(predictions)

    if not silent:
        for i, j in zip(p_names[prediction_order], predictions[prediction_order]):
            print(f"{i:<6}: {j:0.1f}")
        print(" < ".join(p_names[prediction_order]))

    if save_results is not None:
        if not os.path.isdir("results"):
            os.mkdir("results")
        res = dict(zip(p_names, predictions))
        with open(os.path.join("results", model_name + ".json"), "w") as res_file:
            json.dump(res, res_file)

    return predictions


if __name__ == "__main__":
    pass

    """
    models = [
        "esm_single",
        "esm_double",
        "esm_all",
        "af_all",
        "af_single",
        "structures",
        "af_all_f",
        "esm_single_f",
        "af_single_f"
    ]
    par = [2, 3, 4, 5, 6, 7, 8]
    for i in models:
        for p in par:
            print(i, p)
            pos_model = i + "_" + str(p)
            pos_path = os.path.join("saved_models", pos_model + ".pickle")
            if i[-2:] == "_f":
                xi = i[:-2]
            else:
                xi = i
            print(pos_path)
            if os.path.isfile(pos_path):
                predict(pos_model, xi + "_out", p_names, ["769bc", "N0"], True)
            print(" - " * 30)
    
    p_names = np.append(p_names, ["769bc", "N0"])
    temp_cd = np.append(temp_cd, [57.7, 55.6])
    results = np.sort(os.listdir("results"))
    r = []
    rn = []
    for i in results:
        with open(os.path.join("results", i), "r") as res:
            rd = json.load(res)
            r.append(list(map(rd.get, p_names)))
            rn.append(i.split(".")[0])
    r = np.asarray(r)
    dif = r - temp_cd
    fig, ax = plt.subplots(figsize=(18, 18))
    a = ax.imshow(dif, cmap="inferno")
    a.set_clim(-30, 30)
    plt.colorbar(a)
    ax.set_xticks(np.arange(len(p_names)), p_names, rotation=45)
    ax.set_yticks(np.arange(len(rn)), rn)
    dif = dif.round(1)
    for i in range(dif.shape[0]):
        for j in range(dif.shape[1]):
            ax.text(
                j,
                i,
                dif[i, j],
                ha="center",
                va="center",
                size="x-small",
                color="silver",
            )
    fig.tight_layout()
    fig.savefig("test.png")
    """
    
    structs = "af_all"
    pn = np.append(p_names, ["769bc", "N0"])
    temp_cd = np.append(temp_cd, [57.7, 55.6])
    p_inds = np.arange(len(pn))
    for params in [27]: #[1,2,3,4,5,6]:
        error = []
        for i in os.listdir("saved_models"):
            if all(["setting" in i, structs in i, "knn_" not in i]):
                base_split = i.strip().split("_")
                # 2 for LR without prefix
                if base_split[3] == str(params):
                    fit_proteins = np.asarray(base_split[-2].split("-"), dtype=int)
                    test_inds = p_inds[np.invert(np.isin(p_inds, fit_proteins))]
                    pred = predict(
                        "_".join(base_split[:-1]),
                        f"{structs}_out",
                        pn[test_inds],
                        silent=True,
                        loading_method="sk"
                    )
                    err = np.mean(np.abs(pred - temp_cd[test_inds]))
                    error.append(err)
        print(f"Parameter: {params}\nMAE:{np.mean(error)}")
    """
    # structs = "esm_single"
    structs = "af_all"
    # tests = ["4alb", "N2", "N134"]
    tests = ["N5", "N2", "N31"]
    pn = np.append(p_names, ["769bc", "N0"])
    temp_cd = np.append(temp_cd, [57.7, 55.6])
    test_bool = np.isin(pn, tests)
    test_temp = temp_cd[test_bool]
    test_prot = pn[test_bool]
    for i,j in zip(test_prot, test_temp):
        print(f"{i:<6}: {j:0.1f}")
    print("<>"*30)
    par = np.arange(1,5)
    for i in os.listdir("saved_models"):
        if "pickle" in i and structs in i:
            print(i.strip().split(".")[0])
            if not "rf_" in i and "knn_" not in i:
                # predict(i.strip().split(".")[0], f"{structs}_out", p_names)
                pass
            else:
                predict(i.strip().split(".")[0], f"{structs}_out", p_names, loading_method="sk")
    """
