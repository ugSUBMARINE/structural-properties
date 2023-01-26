import os
import sys
import json

import numpy as np
import pandas as pd
import joblib

from regressor_analysis import (
    SB_DATA_DESC,
    HY_DATA_DESC,
    HB_DATA_DESC,
    read_data,
    get_data
)

np.set_printoptions(threshold=sys.maxsize)


def predict(
    model_name: str,
    data_dir: str,
    p_names_in: np.ndarray[tuple[int], np.dtype[str]],
    save_results: bool = False,
    silent: bool = False,
) -> None:
    """predict using a saved model
    :parameter
        - model_name:
          name of the saved model in saved_models/
        - data_dir:
          data of proteins the model should use for predictions
        - p_names_in:
          name of the proteins
        - save_results:
          True to save results under model_name.json in 'results'
        - silent:
          True to not print any output
    :return
        - predictions
          predicted values for all p_names_in
    """
    # reading and calculating data for each protein
    h_bonds_data, hydrophobic_cluster_data, salt_bridges_data = get_data(
        data_dir, p_names_in=p_names_in
    )
    # create DataFrames
    sb_df = pd.DataFrame(
        salt_bridges_data, index=p_names_in, columns=SB_DATA_DESC
    ).round(2)
    hb_df = pd.DataFrame(h_bonds_data, index=p_names_in, columns=HB_DATA_DESC).round(2)
    hy_df = pd.DataFrame(
        hydrophobic_cluster_data, index=p_names_in, columns=HY_DATA_DESC
    ).round(2)

    # make one big DataFrame
    master_frame = pd.concat([hb_df, hy_df, sb_df], axis=1)
    # load data scaler
    scaler = joblib.load(f"{os.path.join('saved_models', f'{model_name}_scaler.save')}")
    inter_master_frame = scaler.transform(master_frame)
    master_frame = pd.DataFrame(
        inter_master_frame, index=master_frame.index, columns=master_frame.columns
    )

    # reading the saved parameters for the model and loading the model
    with open(f"saved_models/{model_name}_setting.txt", "r") as param_file:
        model_param = param_file.readline().strip().split(",")

    data = master_frame.loc[:, model_param]
    model_file_path = f"saved_models/{model_name}.save"
    model = joblib.load(model_file_path)
    predictions = model.predict(data)

    # using the model to predict the data of interest
    prediction_order = np.argsort(predictions)

    if not silent:
        for i, j in zip(p_names_in[prediction_order], predictions[prediction_order]):
            print(f"{i:<6}: {j:0.1f}")
        print(" < ".join(p_names_in[prediction_order]))

    if save_results is not None:
        if not os.path.isdir("results"):
            os.mkdir("results")
        res = dict(zip(p_names_in, predictions))
        with open(os.path.join("results", model_name + ".json"), "w") as res_file:
            json.dump(res, res_file)

    return predictions


if __name__ == "__main__":
    pass
    p_names, _ = read_data()
    predict("test", data_dir="af_all_out", p_names_in=p_names, save_results=True)
