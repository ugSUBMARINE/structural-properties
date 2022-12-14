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
        "MIN  NWS HB",
        "MEAN NWS HB",
        "SUM NWS HB",
        "ICB",
    ]
)

# Index for selcted _DATA_DESC
HB_SELE = [3, 4, 7]
HY_SELE = [0, 3, 7]
SB_SELE = [3, 6, 7]


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
          r?? of each model
        - r2a:
          adjusted r?? of each model
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
    # fit a model to each combinations and calculate the stats of the model
    #  and calculate for each comb the MSE using a LeaveOneOut approach
    r2 = []
    r2a = []
    aics = []
    bics = []
    mses = []
    f_values = []
    models = []
    combs_scores = []
    for i in combs:
        if isinstance(df_list, np.ndarray):
            reg_data = df_list
        else:
            # reshape data as needed
            stacked_df = []
            for di, d in enumerate(df_list):
                stacked_df.append(np.asarray(d[i[di]]).reshape(-1, 1))
            reg_data = np.column_stack(stacked_df)

        # Ordinary Least Square Model
        reg_data = sm.add_constant(reg_data)
        model = sm.OLS(comp_value, reg_data).fit()

        # LeaveOneOut model
        cv = model_selection.LeaveOneOut()
        LOO_model = linear_model.LinearRegression()
        scores = model_selection.cross_val_score(
            LOO_model,
            reg_data,
            comp_value,
            scoring="neg_mean_squared_error",
            cv=cv,
        )

        # store results
        aics.append(model.aic)
        bics.append(model.bic)
        f_values.append(model.fvalue)
        mses.append(model.mse_model)
        r2.append(model.rsquared)
        r2a.append(model.rsquared_adj)
        models.append(model)
        combs_scores.append(np.sqrt(np.mean(np.abs(scores))))

    r2 = np.asarray(r2)
    r2a = np.asarray(r2a)
    aics = np.asarray(aics)
    bics = np.asarray(bics)
    mses = np.asarray(mses)
    f_values = np.asarray(f_values)
    combs_scores = np.asarray(combs_scores)
    return combs_scores, r2, r2a, aics, bics, mses, f_values, models


def use_model(
    comp_value: np.ndarray[tuple[int], np.dtype[int | float]],
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
    # calculate pearson R for the predictions
    predictions = model.predict(np.column_stack((np.ones(len(data)), data)))
    print(f"\nPredicted order:\n{' < '.join(p_names[np.argsort(predictions)])}")
    pr, pp = stats.pearsonr(comp_value, predictions)
    print(f"PearsonR: {pr:>8.4f}\np: {pp:>15.4f}")

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
    cv: list[int | float] | np.ndarray[tuple[int], np.dtype[int | float]],
    save_plot: bool = False,
    show_plot: bool = False,
) -> None:
    """find the model that describes the data the best without overfitting
    :parameter
        - hb_dataframe:
          DataFrame containing all hydrogen bond attributes
        - hy_dataframe:
          DataFrame containing all hydrophobic cluster attributes
        - sb_dataframe:
          DataFrame containing all saltbridges attributes
        - cv:
          experimental values that should be predicted
        - save_plot:
          True to save the plot as performances_plot.png
        - show_plot:
          True to show the scatter plot
    :return
        - None
    """
    print(
        "Order according to experiments:\n"
        f"{' < '.join(p_names[np.argsort(cv)].tolist())}"
    )

    # values of the DataFrames used for fitting
    hb_vals = HB_DATA_DESC[HB_SELE]
    hy_vals = HY_DATA_DESC[HY_SELE]
    sb_vals = SB_DATA_DESC[SB_SELE]

    # make one big DataFrame
    master_frame = pd.concat(
        [hb_dataframe[hb_vals], hy_dataframe[hy_vals], sb_dataframe[sb_vals]],
        axis=1,
    )
    # make all single, double, ... NumMastervals combinations and test their performance
    master_vals = list(hb_vals) + list(hy_vals) + list(sb_vals)
    used_combinations = []
    performances_combinations = []
    for i in range(1, len(master_vals) + 1):
        comb_i = list(itertools.combinations(master_vals, i))
        for c in comb_i:
            res = analysis(cv, [c], np.asarray(master_frame.loc[:, c]))
            performances_combinations.append([i[0] for i in res])
            used_combinations.append(c)
    performances_combinations = np.asarray(performances_combinations)
    used_combinations = np.asarray(used_combinations, dtype=object)

    # index of return of analysis
    performance_criteria = 0

    print("\nBest 10 combinations:")
    for i in used_combinations[
        np.argsort(performances_combinations[:, performance_criteria])
    ][:10]:
        print(f"{' - '.join(i)}")
    print("\nBest 10 combinations (Values are MSEs of LOO model):")
    for i in performances_combinations[:, 0][
        np.argsort(performances_combinations[:, performance_criteria])
    ][:10]:
        print(f"{i:0.4f}")
    print("\nBest 10 combinations (Values are r?? from OLS):")
    for i in performances_combinations[:, 1][
        np.argsort(performances_combinations[:, performance_criteria])
    ][:10]:
        print(f"{i:0.4f}")

    best_comp = np.argmin(performances_combinations[:, performance_criteria])
    use_model(
        cv,
        performances_combinations[:, -1][best_comp],
        np.asarray(master_frame.loc[:, used_combinations[best_comp]]),
        save_plot,
        show_plot,
    )


if __name__ == "__main__":
    pass
