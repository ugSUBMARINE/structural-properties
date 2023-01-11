import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from sklearn import linear_model, model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import joblib
import statsmodels.api as sm

from run_rmsf_analysis import p_names, temp_cd, activity
from properties import HydrophobicClusterOwn, SaltBridges

np.set_printoptions(threshold=sys.maxsize)
plt.style.use("default")
plt.rcParams.update({"font.size": 25})
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)


SB_DATA_DESC = np.asarray(
    [
        "MAX IA SB",
        "MIN IA SB",
        "MEAN IA SB",
        "SUM IA SB",
        "MAX NWS SB",
        "MIN NWS SB",
        "MEAN NWS SB",
        "SUM NWS SB",
        "ICB SB",
    ]
)
HY_DATA_DESC = np.asarray(
    [
        "MAX CA",
        "MIN CA",
        "MEAN CA",
        "SUM CA",
        "MAX CC",
        "MIN CC",
        "MEAN CC",
        "SUM CC",
        "ICH",
    ]
)
HB_DATA_DESC = np.asarray(
    [
        "MAX BPN HB",
        "MIN BPN HB",
        "MEAN BPN HB",
        "SUM BPN HB",
        "MAX NWS HB",
        "MIN NWS HB",
        "MEAN NWS HB",
        "SUM NWS HB",
        "ICB",
    ]
)


def spinning_wheele():
    """uses itertools to be able to take the next sign with 'next' to create a spinning
    wheele
    :parameter
        - None:
    :return
        - None
    """
    return itertools.cycle(["-", "\\", "|", "/"])


def lr_analysis(
    comp_value: np.ndarray[tuple[int], np.dtype[float]],
    df_list: list[pd.DataFrame],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[float]],
    np.ndarray[tuple[int], np.dtype[float]],
    np.ndarray[tuple[int], np.dtype[float]],
    np.ndarray[tuple[int], np.dtype[float]],
    np.ndarray[tuple[int], np.dtype[float]],
    np.ndarray[tuple[int], np.dtype[float]],
    np.ndarray[tuple[int], np.dtype[float]],
    np.ndarray[tuple[int], np.dtype[float]],
]:
    """fit models to and retrieve their performances
    :parameter
        - comp_value:
          values the calculated values should be fit to
        - df_list:
          list of DataFrames with the set of values the model should be fitted to
    :return
        - combs_scores:
          MSE of each model based on LeaveOneOut
        - r2:
          r² of each model
        - r2a:
          adjusted r² of each model
        - aics:
          Akaike's information criteria of each models
        - bics:
          Bayes' information criteria of each model
        - mses:
          MSE of each model
        - f_values:
          F-statistic of the fully specified model for each model
        - models:
          statsmodels regression model instance of each model
    """

    # Ordinary Least Square Model
    df_list = sm.add_constant(df_list)
    model = sm.OLS(comp_value, df_list).fit()

    # LeaveOneOut model
    cv = model_selection.LeaveOneOut()
    LOO_model = linear_model.LinearRegression()
    scores = model_selection.cross_val_score(
        LOO_model,
        df_list,
        comp_value,
        scoring="neg_mean_squared_error",
        cv=cv,
    )

    # store results
    aics = np.asarray(model.aic)
    bics = np.asarray(model.bic)
    f_values = np.asarray(model.fvalue)
    mses = np.asarray(model.mse_model)
    r2 = np.asarray(model.rsquared)
    r2a = np.asarray(model.rsquared_adj)
    combs_scores = np.asarray(np.sqrt(np.mean(np.abs(scores))))
    return combs_scores, r2, r2a, aics, bics, mses, f_values, model


def rf_analysis(
    comp_value: np.ndarray[tuple[int], np.dtype[float]], df_list: list[pd.DataFrame]
) -> tuple[float, float, RandomForestRegressor]:
    """fit RandomForestRegressor to data
    :parameter
        - comp_value:
          values the calculated values should be fit to
        - df_list:
          list of DataFrames with the set of values the model should be fitted to
    :return
        - 1 - oob_score_
          out of bag r2 of all trees
        - mae
          mean absolute prediction error
        - regr
          RandomForestRegressor
    """
    regr = RandomForestRegressor(
        max_depth=3, random_state=0, n_estimators=10, oob_score=True, n_jobs=10
    )
    regr.fit(df_list, comp_value)
    mae = np.mean(np.abs(regr.oob_prediction_ - comp_value))
    return 1 - regr.oob_score_, mae, regr


def knn_analysis(
    comp_value: np.ndarray[tuple[int], np.dtype[float]],
    df_list: list[pd.DataFrame],
    nneighbors: int = 2,
) -> tuple[float, float, KNeighborsRegressor]:
    """fit KNeighborsRegressor to data with LeaveOneOut
    :parameter
        - comp_value:
          values the calculated values should be fit to
        - df_list:
          list of DataFrames with the set of values the model should be fitted to
    :return
        - combs_scores:
          MSE of each model based on LeaveOneOut
        - mae
          mean absolute prediction error
        - knn
          KNeighborsRegressor
    """
    knn = KNeighborsRegressor(n_neighbors=nneighbors, weights="distance", n_jobs=10)
    knn.fit(df_list, comp_value)

    # LeaveOneOut model
    cv = model_selection.LeaveOneOut()
    LOO_model = KNeighborsRegressor(
        n_neighbors=nneighbors, weights="distance", n_jobs=10
    )
    scores = model_selection.cross_val_score(
        LOO_model,
        df_list,
        comp_value,
        scoring="neg_mean_squared_error",
        cv=cv,
    )
    combs_scores = np.asarray(np.sqrt(np.mean(np.abs(scores))))
    mae = np.mean(np.abs(knn.predict(df_list) - comp_value))
    return combs_scores, mae, knn


def create_best_model(
    hb_dataframe: pd.DataFrame,
    hy_dataframe: pd.DataFrame,
    sb_dataframe: pd.DataFrame,
    hb_vals: list | np.ndarray[tuple[int], np.dtype[str]],
    hy_vals: list | np.ndarray[tuple[int], np.dtype[str]],
    sb_vals: list | np.ndarray[tuple[int], np.dtype[str]],
    cv: list[int | float] | np.ndarray[tuple[int], np.dtype[int | float]],
    p_names: list[str],
    save_plot: bool = False,
    show_plot: bool = False,
    save_model: str | None = None,
    chose_model_ind: int | None = None,
    force_cmi: bool = False,
    force_np: int | None = None,
    regressor: str = "LR",
) -> None:
    """find the model that describes the data the best without over fitting
    :parameter
        - hb_dataframe:
          DataFrame containing all hydrogen bond attributes
        - hy_dataframe:
          DataFrame containing all hydrophobic cluster attributes
        - sb_dataframe:
          DataFrame containing all saltbridges attributes
        - cv:
          experimental values that should be predicted
        - p_names:
          name of the proteins
        - save_plot:
          True to save the plot as performances_plot.png
        - show_plot:
          True to show the scatter plot
        - save_model:
          to store model use filepath where the model should be stored
        - chose_model_ind:
          index of the 10 best models that should be used (and saved)
        - force_cmi:
          if chose_model_ind is not under the best 10 - no model gets returned
        - force_np:
          force number of explored parameters to be of force_np and
          not from 1 to sum(*_param_num)
        - regressor:
          *  'LR' for linear regression
          *  'RF' for random forest
          *  'KNN' for k-nearest neighbors
    :return
        - None
    """
    print(
        "Order according to experiments:\n"
        f"{' < '.join(p_names[np.argsort(cv)].tolist())}"
    )

    # make one big DataFrame
    master_frame = pd.concat([hb_dataframe, hy_dataframe, sb_dataframe], axis=1)
    # make all single, double, ... NumMastervals combinations and test their performance
    master_vals = list(hb_vals) + list(hy_vals) + list(sb_vals)

    # set sizes of combinations to test
    num_mv = len(master_vals)
    start = 1
    if force_np is not None:
        start = force_np
        num_mv = force_np

    if regressor == "LR":
        analysis = lr_analysis
    elif regressor == "RF":
        analysis = rf_analysis
    elif regressor == "KNN":
        analysis = knn_analysis
    else:
        raise KeyError(f"Invalid regressor encountered: '{regressor}'")

    # fit regressor to all possible attribute combinations and store the(m) + results
    used_combinations = []
    performances_combinations = []
    spinner = spinning_wheele()
    for i in range(start, num_mv + 1):
        comb_i = list(itertools.combinations(master_vals, i))
        for ci, c in enumerate(comb_i):
            if ci % 25 == 0:
                print("\r" + next(spinner), end="")
            res = analysis(cv, np.asarray(master_frame.loc[:, c]))
            performances_combinations.append(list(res))
            used_combinations.append(c)
    print("\r", end="")
    performances_combinations = np.asarray(performances_combinations)
    used_combinations = np.asarray(used_combinations, dtype=object)

    # index of return of analysis
    performance_criteria = 0
    performance_order = np.argsort(performances_combinations[:, performance_criteria])

    print("\nBest 10 combinations:")
    for i in used_combinations[performance_order][:10]:
        print(f"{' - '.join(i)}")
    print("\nBest 10 combinations (Values are MSEs of LOO model):")
    for i in performances_combinations[:, 0][performance_order][:10]:
        print(f"{i:0.4f}")
    print("\nBest 10 combinations (Values are r² from OLS):")
    for i in performances_combinations[:, 1][performance_order][:10]:
        print(f"{i:0.4f}")

    # how to determine the best performance
    best_comp = performance_order[0]
    if chose_model_ind is not None:
        if type(chose_model_ind) == int:
            best_comp = performance_order[chose_model_ind]
        elif type(chose_model_ind) == str:
            if chose_model_ind.startswith("!"):
                # desired number of parameters
                des_num_param = int(chose_model_ind[1:])
            else:
                des_num_param = int(chose_model_ind)
            params = np.asarray(
                [len(i) for i in used_combinations[performance_order][:10]]
            )
            # model with the closest number of parameters of the top 10 models
            param_diff = np.abs(params - des_num_param)
            best_comp = performance_order[np.argmin(param_diff)]
            if force_cmi and np.min(param_diff) != 0:
                print("Failed to find model with expected number of parameters")
                return
    print(f"Chosen combination: {used_combinations[best_comp]}")

    # calculate Pearson R for the predictions
    model = performances_combinations[:, -1][best_comp]
    if regressor == "LR":
        predictions = model.predict(
            np.column_stack(
                (
                    np.ones(
                        len(
                            np.asarray(
                                master_frame.loc[:, used_combinations[best_comp]]
                            )
                        )
                    ),
                    np.asarray(master_frame.loc[:, used_combinations[best_comp]]),
                )
            )
        )
    else:
        predictions = model.predict(
            np.asarray(master_frame.loc[:, used_combinations[best_comp]])
        )
    print(f"\nPredicted order:\n{' < '.join(p_names[np.argsort(predictions)])}")
    pr, pp = stats.pearsonr(cv, predictions)
    mae = np.mean(np.abs(cv - predictions))
    print(f"PearsonR: {pr:>17.4f}\np: {pp:>24.4f}")
    print(f"Mean Absolute Error: {mae:0.4f}")

    # scatter plot of ground truth (x) and predictions (y)
    fig, ax = plt.subplots(figsize=(32, 18))
    for i in range(len(predictions)):
        ax.scatter(cv[i], predictions[i], label=p_names[i])
    fig.legend(loc="lower center", ncol=10)
    ax.set(ylabel="predicted values", xlabel="ground truth")
    if save_plot:
        fig.savefig("performances_plot.png")
    if show_plot:
        plt.show()

    # save the model
    if save_model is not None:
        if not os.path.isdir("saved_models"):
            os.mkdir("saved_models")
        sett = open(os.path.join("saved_models", save_model + "_setting.txt"), "w+")
        sett.write(",".join(list(used_combinations[best_comp])))
        sett.close()
        model_path = os.path.join("saved_models", save_model + ".pickle")
        if regressor == "LR":
            performances_combinations[:, -1][best_comp].save(model_path)
        else:
            joblib.dump(performances_combinations[:, -1][best_comp], model_path)


if __name__ == "__main__":
    pass
