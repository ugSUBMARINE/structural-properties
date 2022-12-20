import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from sklearn import linear_model, model_selection
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


def analysis(
    comp_value: np.ndarray[tuple[int], np.dtype[float]],
    combs: list[tuple[str]],
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
        - combs
          combinations of names of different columns of the DataFrames
          in the same order as the DataFrames
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
          F-statistic of the fully specified model for each models
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


def use_model(
    comp_value: np.ndarray[tuple[int], np.dtype[int | float]],
    p_names: list[str],
    model: sm.OLS,
    data: np.ndarray[tuple[int, int], np.dtype[int | float]],
    save_plot: bool = False,
    show_plot: bool = False,
) -> None:
    """uses the fitted models to calculate the comp_value based on their inputs that
    were used to fit them and plot the correlation plot
    :parameter
        - comp_value:
          values the calculated values should be fit to
        - p_names:
          name of the proteins
        - model:
          statsmodels linear_model instance
        - data:
          data used to fit the model
        - save_plot:
          True to save the plot as performances_plot.png
        - show_plot:
          True to show the scatter plot
    :return
        - func1return
          description
    """
    # calculate Pearson R for the predictions
    predictions = model.predict(np.column_stack((np.ones(len(data)), data)))
    print(f"\nPredicted order:\n{' < '.join(p_names[np.argsort(predictions)])}")
    pr, pp = stats.pearsonr(comp_value, predictions)
    print(f"PearsonR: {pr:>17.4f}\np: {pp:>24.4f}")
    print(f"Mean Absolute Error: {np.mean(np.abs(comp_value - predictions)):0.4f}")

    # scatter plot of ground truth (x) and predictions (y)
    fig, ax = plt.subplots(figsize=(32, 18))
    for i in range(len(predictions)):
        ax.scatter(comp_value[i], predictions[i], label=p_names[i])
    fig.legend(loc="lower center", ncol=10)
    ax.set(ylabel="predicted values", xlabel="ground truth")
    if save_plot:
        fig.savefig("performances_plot.png")
    if show_plot:
        plt.show()


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
          index of the 10 best models that should be used (and saved )
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
    used_combinations = []
    performances_combinations = []
    for i in range(1, len(master_vals) + 1):
        comb_i = list(itertools.combinations(master_vals, i))
        for c in comb_i:
            res = analysis(cv, [c], np.asarray(master_frame.loc[:, c]))
            performances_combinations.append(list(res))
            used_combinations.append(c)
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
            best_comp = performance_order[np.argmin(np.abs(params - des_num_param))]
    print(f"Chosen combination: {used_combinations[best_comp]}")
    use_model(
        cv,
        p_names,
        performances_combinations[:, -1][best_comp],
        np.asarray(master_frame.loc[:, used_combinations[best_comp]]),
        save_plot,
        show_plot,
    )
    if save_model is not None:
        if not os.path.isdir("saved_models"):
            os.mkdir("saved_models")
        sett = open(os.path.join("saved_models", save_model + "_setting.txt"), "w+")
        sett.write(",".join(list(used_combinations[best_comp])))
        sett.close()
        performances_combinations[:, -1][best_comp].save(
            os.path.join("saved_models", save_model + ".pickle")
        )


if __name__ == "__main__":
    pass
