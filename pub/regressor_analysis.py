import os
import sys
import itertools
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib

from run_rmsf_analysis import p_names, temp_cd, activity
from properties import SaltBridges, HydrophobicClusterOwn


np.set_printoptions(threshold=sys.maxsize)
np.random.seed(42)
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


def spinning_wheele() -> None:
    """uses itertools to be able to take the next sign with 'next' to create a spinning
    wheel
    :parameter
        - None:
    :return
        - None
    """
    return itertools.cycle(["-", "\\", "|", "/"])


def single_stats(
    desc: list[str],
    property_data: np.ndarray[tuple[int, int], np.dtype[int | float]],
    cv: list[int | float] | np.ndarray[tuple[int], np.dtype[int | float]],
    num_params: int = 3,
    silent: bool = False,
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
        - silent:
          True to hide output in terminal
    :return
        - chosen
          num_params *_DATA_DESC with the highest PearsonR
    """
    descriptions, pr, pp = [], [], []
    for i in range(len(desc)):
        i_data = property_data[:, i]
        if len(np.unique(i_data)) > 1:
            ipr, ipp = stats.pearsonr(cv, i_data)
            if not silent:
                print(f"--- {desc[i]} ---")
                print(
                    f"PearsonR: {ipr:>8.4f}\np: {ipp:>15.4f}\n"
                    f"std: {i_data.std():13.4f}"
                )
                print(f"Max: {np.max(i_data):13.4f}\nMin: {np.min(i_data):13.4f}")
            descriptions.append(desc[i])
            pr.append(ipr)
            pp.append(ipp)
    order = np.lexsort((pp, -np.abs(pr)))
    chosen = np.asarray(descriptions)[order][:num_params]
    if not silent:
        print(f"Chosen vals: {' / '.join(chosen.tolist())}")
    return chosen


def get_data(
    data_path: str,
    hb_end: str = "_hb",
    hy_end: str = "_hy",
    sb_end: str = "_sb",
    file_type: str = "csv",
    p_names_in: np.ndarray[tuple[int], np.dtype[str]] | None = None,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[float]],
    np.ndarray[tuple[int, int], np.dtype[float]],
    np.ndarray[tuple[int, int], np.dtype[float]],
]:
    """read and process data of attribute output files
    :parameter
        - data_path:
          file path where all files are stored
        - hb_end, hy_end, sb_end:
          file suffix for H-Bonds, hydrophobic cluster and salt bridges outfiles
        - p_names_in:
          name of the proteins in their files
    :return
        - data of read H-Bonds, hydrophobic cluster and salt bridges outfiles
          description
    """

    # read and process data of every protein in p_names
    hb_data = []
    hy_data = []
    sb_data = []
    for i in p_names_in:
        hb_path = f"{os.path.join(f'{data_path}',f'{i}{hb_end}.{file_type}')}"
        hb = SaltBridges(hb_path, -1, HB_DATA_DESC)
        hb_data.append(list(hb.stats()[:-1]))
        hy_path = f"{os.path.join(f'{data_path}',f'{i}{hy_end}.{file_type}')}"
        hy = HydrophobicClusterOwn(hy_path, HY_DATA_DESC)
        hy_data.append(list(hy.stats())[:-1])
        sb_path = f"{os.path.join(f'{data_path}',f'{i}{sb_end}.{file_type}')}"
        sb = SaltBridges(sb_path, -1, SB_DATA_DESC)
        sb_data.append(list(sb.stats())[:-1])
    return np.asarray(hb_data), np.asarray(hy_data), np.asarray(sb_data)


def cross_val(
    predictor,
    X: pd.DataFrame,
    y: np.ndarray[tuple[int], np.dtype[int | float]],
    k: int = 5,
    fi: bool = False,
    parallelize: int | None = None,
) -> tuple[float, float, np.ndarray[tuple[int, int], np.dtype[float]]]:
    """cross validation as k-fold or LeaveOneOut (if k == n)
    :parameter
        - predictor:
          a sklearn predictor object
        - X:
          the data that is used to fit
        - y:
          target X should be fit to
        - k:
          number of splits of X for cross validation
        - fi:
          whether feature importance of RandomForestRegressor should be saved
        - parallelize:
          None to not parallelize the cross validation, integer to specify the number
          of cores or '-1' to use all cores
    :return
        - mae:
          mean absolute error over all predictions in the test sets
        - r2:
          r2 over all predictions in the test sets
        - feature_imp:
          feature importance from RandomForestRegressor
    """

    def cv_parallel(
        fold: np.ndarray[tuple[int], np.dtype[int | float]],
        p_x: pd.DataFrame,
        p_y: np.ndarray[tuple[int], np.dtype[int | float]],
        pp_inds: list[int] | np.ndarray[tuple[int], np.dtype[int]],
        fi: bool,
    ):
        # data points in split fold
        i_cv_bool = np.isin(pp_inds, fold)
        # data points not in split fold
        fit_p = np.invert(i_cv_bool)
        # predictions
        res = predictor.fit(p_x.loc[fit_p], p_y[fit_p])
        # predictions and ground truth
        par_pred_ = res.predict(p_x.loc[i_cv_bool])
        par_gt_ = p_y[i_cv_bool]
        # feature importances of RandomForestRegressor
        if fi:
            par_fi_ = res.feature_importances_
        else:
            par_fi_ = None
        return par_pred_, par_gt_, par_fi_

    # indices of all data points
    p_inds = np.arange(len(X))
    np.random.shuffle(p_inds)
    # k splits
    c_folds = np.array_split(p_inds, k)
    if parallelize is None:
        predictions = []
        ground_truth = []
        # feature importance of RandomForestRegressor
        feature_imp = []
        for i in c_folds:
            # data points in split i
            i_cv_bool = np.isin(p_inds, i)
            # data points not in split i
            fit_p = np.invert(i_cv_bool)
            # predictions
            res = predictor.fit(X.loc[fit_p], y[fit_p])
            predictions += res.predict(X.loc[i_cv_bool]).tolist()
            ground_truth += y[i_cv_bool].tolist()
            if fi:
                feature_imp.append(res.feature_importances_)
        predictions = np.asarray(predictions)
        ground_truth = np.asarray(ground_truth)
    else:
        # Parallel execution
        predictions, ground_truth, feature_imp = zip(
            *np.asarray(
                joblib.Parallel(n_jobs=parallelize)(
                    joblib.delayed(cv_parallel)(f, X, y, p_inds, fi) for f in c_folds
                ),
                dtype=object,
            )
        )
        # concatenate the outputs
        inter_pred = []
        inter_gt = []
        inter_feature_imp = []
        for i in range(len(predictions)):
            inter_pred += predictions[i].tolist()
            inter_gt += ground_truth[i].tolist()
            inter_feature_imp.append(feature_imp[i])

        predictions = np.asarray(inter_pred)
        ground_truth = np.asarray(inter_gt)
        feature_imp = np.asarray(inter_feature_imp)

    # mean absolute error and r2 of all test data points
    mae = np.mean(np.abs(predictions - ground_truth))
    r2 = r2_score(ground_truth, predictions)
    # mean feature importance over all k- folds
    if fi:
        feature_imp = np.mean(feature_imp, axis=0)
    else:
        feature_imp = [None]
    return mae, r2, feature_imp


def fit_data(
    data_path,
    model_ind: int | str | None = None,
    save_model: str | None = None,
    target: np.ndarray[tuple[int], np.dtype[int | float]] | None = None,
    hb_param_num: int = 3,
    hy_param_num: int = 3,
    sb_param_num: int = 3,
    ow_hb_vals: np.ndarray[tuple[int], np.dtype[str]] | None = None,
    ow_hy_vals: np.ndarray[tuple[int], np.dtype[str]] | None = None,
    ow_sb_vals: np.ndarray[tuple[int], np.dtype[str]] | None = None,
    p_names_in: np.ndarray[tuple[int], np.dtype[str]] | None = None,
    chose_model_ind: int | None = None,
    force_cmi: bool = False,
    force_np: int | None = None,
    explore_all: bool = False,
    regressor: str = "LR",
    c_val: int | None = 5,
    silent: bool = False,
    paral: int | None = None,
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
        - target:
          data to fit the model to
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
        - chose_model_ind:
          index of the 10 best models that should be used (and saved)
        - force_cmi:
          if chose_model_ind is not under the best 10 - no model gets returned
        - force_np:
          force number of explored parameters to be of force_np and
          not from 1 to sum(*_param_num)
        - explore_all:
          True to ignore *_vals and use all *_DATA_DESC to find best parameter combi
        - regressor:
          *  'LR' for linear regression
          *  'RF' for random forest
          *  'KNN' for k-nearest neighbors
          *  'RI' for Ridge
          *  'GB' for GradientBoostingRegressor
        - c_val:
          integer do specify the number of splits or
          None for LeaveOneOut cross validation
        - silent:
          True to hide output in terminal
        - paral:
          None to not parallelize the cross validation, integer to specify the number
          of cores or '-1' to use all cores
    :return
        - None
    """

    # retrieve the data
    h_bonds_data, hydrophobic_cluster_data, salt_bridges_data = get_data(
        data_path, p_names_in=p_names_in
    )

    if explore_all:
        hb_vals = HB_DATA_DESC
        hy_vals = HY_DATA_DESC
        sb_vals = SB_DATA_DESC
    else:
        if not silent:
            print("Hydrogen Bonds")
        hb_vals = single_stats(HB_DATA_DESC, h_bonds_data, target, hb_param_num, silent)
        if not silent:
            print("\nHydrophobic Cluster")
        hy_vals = single_stats(
            HY_DATA_DESC, hydrophobic_cluster_data, target, hy_param_num, silent
        )
        if not silent:
            print("\nSalt Bridges")
        sb_vals = single_stats(
            SB_DATA_DESC, salt_bridges_data, target, sb_param_num, silent
        )

    # to be able to overwrite the *_vals
    if ow_hb_vals is not None:
        hb_vals = ow_hb_vals
    if ow_hy_vals is not None:
        hy_vals = ow_hy_vals
    if ow_sb_vals is not None:
        sb_vals = ow_sb_vals

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
    # scale values so each feature is in range (0, 1)
    # """
    scaler = MinMaxScaler()
    scaler.fit(master_frame)
    inter_master_frame = scaler.transform(master_frame)
    master_frame = pd.DataFrame(
        inter_master_frame, index=master_frame.index, columns=master_frame.columns
    )
    # """
    # make all single, double, ... NumMastervals combinations and test their performance
    master_vals = list(hb_vals) + list(hy_vals) + list(sb_vals)

    # whether feature importance is possible
    get_fi = False
    # set regressor
    if regressor == "LR":
        if not silent:
            print("Chosen Regressor: LinearRegression")
        LOO_model = LinearRegression()
    elif regressor == "RI":
        if not silent:
            print("Chosen Regressor: Ridge")
        LOO_model = Ridge()
    elif regressor == "RF":
        if not silent:
            print("Chosen Regressor: RandomForestRegressor")
        get_fi = True
        LOO_model = RandomForestRegressor(
            max_depth=3, random_state=0, n_estimators=10, oob_score=True, n_jobs=10
        )
    elif regressor == "KNN":
        if not silent:
            print("Chosen Regressor: KNeighborsRegressor")
        LOO_model = KNeighborsRegressor(n_neighbors=2, weights="distance", n_jobs=10)
    elif regressor == "GB":
        if not silent:
            print("Chosen Regressor: GradientBoostingRegressor")
        get_fi = True
        LOO_model = GradientBoostingRegressor(n_estimators=10, random_state=0)
    else:
        raise KeyError(f"Invalid regressor encountered: '{regressor}'")

    # set sizes of combinations to test
    num_mv = len(master_vals)
    start = 1
    if force_np is not None:
        start = force_np
        num_mv = force_np

    # set size of validation set for cross validation
    if c_val is None:
        split = master_frame.shape[0]
    else:
        split = c_val

    # fit regressor to all possible attribute combinations and store the(m) + results
    used_combinations = []
    mae = []
    r2 = []
    # feature importances
    fis = []
    spinner = spinning_wheele()
    for i in range(start, num_mv + 1):
        comb_i = list(itertools.combinations(master_vals, i))
        for ci, c in enumerate(comb_i):
            # spin wheel while working
            if ci % 25 == 0:
                print("\r" + next(spinner), end="")
            # cross validation
            i_mae, i_r2, i_fi = cross_val(
                LOO_model, master_frame.loc[:, c], target, split, get_fi, paral
            )
            mae.append(i_mae)
            r2.append(i_r2)
            fis.append(i_fi)
            used_combinations.append(c)
    mae = np.asarray(mae)
    r2 = np.asarray(r2)
    fis = np.asarray(fis)
    used_combinations = np.asarray(used_combinations, dtype=object)

    performance_order = np.argsort(mae)
    if not silent:
        print("\nBest 10 combinations:")
        for i in used_combinations[performance_order][:10]:
            print(f"{' - '.join(i)}")
        print("\nBest 10 combinations (MSEs of LOO model):")
        for i in mae[performance_order][:10]:
            print(f"{i:0.4f}")
        print("\nBest 10 combinations (r² over all LOO predictions):")
        for i in r2[performance_order][:10]:
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
    if not silent:
        print(f"\nChosen combination: {used_combinations[best_comp]}")

    # fit model to data
    fit_model = LOO_model.fit(master_frame.loc[:, c], target)
    # save fitted model
    if save_model is not None:
        if not os.path.isdir("saved_models"):
            os.mkdir("saved_models")
        sett = open(os.path.join("saved_models", save_model + "_setting.txt"), "w+")
        sett.write(",".join(list(used_combinations[best_comp])))
        sett.close()
        model_path = os.path.join("saved_models", save_model + ".pickle")
        joblib.dump(fit_model, model_path)

    return mae, r2, fis, used_combinations, fit_model


def plot_search(
    best_errors: np.ndarray[tuple[int], np.dtype[int | float]],
    performance_order: np.ndarray[tuple[int], np.dtype[int | float]],
    order: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(32, 18))
    x_ = np.arange(len(best_errors))
    # mark best
    ax.scatter(
        x_[performance_order],
        best_errors[performance_order],
        color="firebrick",
        s=150,
        marker="^",
    )
    # error course
    ax.plot(best_errors, marker="o", color="forestgreen")
    # reorder labels
    if order == -1:
        x_ = x_[::-1]
    ax.set_xticks(x_, np.arange(1, len(best_errors) + 1))
    ax.set_xlabel("number of attributes")
    ax.set_ylabel("cross validation error")
    plt.show()


class AttributeSearch:
    def __init__(
        self,
        regressor: str,
        target: np.ndarray[tuple[int], np.dtype[int | float]],
        p_names_in: np.ndarray[tuple[int], np.dtype[int | float]],
        structs_in: str,
        hb_vals: np.ndarray[tuple[int], np.dtype[str]],
        hy_vals: np.ndarray[tuple[int], np.dtype[str]],
        sb_vals: np.ndarray[tuple[int], np.dtype[str]],
        c_val: int | None = None,
        silent: bool = False,
        paral: int | None = None,
        plot_search_res: bool = True,
    ):
        """Different search functions to find the best attribute set
        :parameter
            - regressor:
              *  'LR' for linear regression
              *  'RF' for random forest
              *  'KNN' for k-nearest neighbors
              *  'RI' for Ridge
              *  'GB' for GradientBoostingRegressor
            - target:
              data to fit the model to
            - p_names_in:
              Name of the proteins and their respective files like 769bc for 769bc.pdb
            - hb_vals:
              H-Bonds attributes to be tested
            - hy_vals:
              Hydrophobic Cluster attributes to be tested
            - sb_vals:
              Salt Bridges attributes to be tested
            - c_val:
              integer do specify the number of splits or
              None for LeaveOneOut cross validation
            - silent:
              True to hide output in terminal
            - paral:
              None to not parallelize the cross validation,
              integer to specify the number of cores or '-1' to use all cores
            - plot_search_res:
              whether the course of the error over the search should be plotted
        :return
            - None
        """
        self.regressor = regressor
        self.target = target
        self.p_names_in = p_names_in
        self.structs_in = structs_in
        self.hb_vals = hb_vals
        self.hy_vals = hy_vals
        self.sb_vals = sb_vals
        self.c_val = c_val
        self.silent = silent
        self.paral = paral
        self.plot_search_res = plot_search_res

    def forward_search(self):
        """
        tries to find the best attribute combination by greedy adding the attribute
        that yields in the lowest error
        :parameters
            - None
        :return
            - None
        """
        new_hb_vals = self.hb_vals
        new_hy_vals = self.hy_vals
        new_sb_vals = self.sb_vals
        conc_vals = np.concatenate((new_hb_vals, new_hy_vals, new_sb_vals))
        # total number of attributes
        total_num_vals = conc_vals.shape[0]
        # the so far best attribute addings
        best_vals = []
        # all errors per added best attribute
        best_errors = []
        for i in range(total_num_vals):
            best_c = None
            best_err = np.inf
            # search over each attribute that's not added so far
            for c in conc_vals[np.invert(np.isin(conc_vals, best_vals))]:
                added_val = np.append(best_vals, c)
                new_hb_vals = added_val[np.isin(added_val, self.hb_vals)]
                new_hy_vals = added_val[np.isin(added_val, self.hy_vals)]
                new_sb_vals = added_val[np.isin(added_val, self.sb_vals)]
                mae_f, r2_f, fis_f, used_combinations_f, fit_model_f = fit_data(
                    f"{self.structs_in}_out",
                    force_np=added_val.shape[0],
                    explore_all=True,
                    p_names_in=self.p_names_in,
                    target=self.target,
                    regressor=self.regressor,
                    silent=self.silent,
                    ow_hb_vals=new_hb_vals,
                    ow_hy_vals=new_hy_vals,
                    ow_sb_vals=new_sb_vals,
                    paral=self.paral,
                    c_val=self.c_val,
                )
                # update best added attribute if error improves
                if best_err > mae_f[0]:
                    best_err = mae_f[0]
                    best_c = c
            best_vals.append(best_c)
            best_errors.append(best_err)

        best_vals = np.asarray(best_vals)
        best_errors = np.asarray(best_errors)

        # determine the best attribute combination and plot the cource of the search
        performance_order = np.argmin(best_errors)
        best_comb_vals = best_vals[: performance_order + 1]
        print(
            f"Best combination of attributes: {' - '.join(best_comb_vals.tolist())}\n"
            f"MAE: {best_errors[performance_order]:.4f}"
        )
        if self.plot_search_res:
            plot_search(best_errors, performance_order)

    def backward_search(self):
        """
        tries to find the best attribute combination by greedy removing the attribute
        whichs removal yields the lowest errors
        :parameters
            - None
        :return
            - None
        """
        new_hb_vals = self.hb_vals
        new_hy_vals = self.hy_vals
        new_sb_vals = self.sb_vals
        conc_vals = np.concatenate((new_hb_vals, new_hy_vals, new_sb_vals))
        # total number of attributes
        total_num_vals = conc_vals.shape[0]
        # the so far best attribute addings
        best_vals = []
        # all errors per added best attribute
        best_errors = []
        for i in range(total_num_vals-1):
            best_c = None
            best_err = np.inf
            # search over each attribute that's not added so far
            for c in conc_vals:
                i_cv = conc_vals[conc_vals != c]
                new_hb_vals = i_cv[np.isin(i_cv, self.hb_vals)]
                new_hy_vals = i_cv[np.isin(i_cv, self.hy_vals)]
                new_sb_vals = i_cv[np.isin(i_cv, self.sb_vals)]
                mae_f, r2_f, fis_f, used_combinations_f, fit_model_f = fit_data(
                    f"{self.structs_in}_out",
                    force_np=i_cv.shape[0],
                    explore_all=True,
                    p_names_in=self.p_names_in,
                    target=self.target,
                    regressor=self.regressor,
                    silent=self.silent,
                    ow_hb_vals=new_hb_vals,
                    ow_hy_vals=new_hy_vals,
                    ow_sb_vals=new_sb_vals,
                    paral=self.paral,
                    c_val=self.c_val,
                )
                # update best added attribute if error improves
                if best_err > mae_f[0]:
                    best_err = mae_f[0]
                    best_c = i_cv
            conc_vals = best_c
            best_vals.append(best_c)
            best_errors.append(best_err)


        # determine the best attribute combination and plot the cource of the search
        performance_order = np.argmin(best_errors)
        best_comb_vals = best_vals[performance_order]
        print(
            f"Best combination of attributes: {' - '.join(best_comb_vals)}\n"
            f"MAE: {best_errors[performance_order]:.4f}"
        )
        if self.plot_search_res:
            plot_search(best_errors, performance_order, -1)

    def model_based_search(self):
        """
        uses the coef_ or feature_importances_ of a model to iteratively remove the
        attribute with the lowest importances
        :parameters
            - None
        :return
            - None
        """
        new_hb_vals = self.hb_vals
        new_hy_vals = self.hy_vals
        new_sb_vals = self.sb_vals
        conc_vals = np.concatenate((new_hb_vals, new_hy_vals, new_sb_vals))
        # total number of attributes
        total_num_vals = conc_vals.shape[0]
        # all attribute removals
        best_vals = []
        # all errors per added best attribute
        best_errors = []
        # attribute importances
        att_imp = []
        for i in range(total_num_vals):
            mae_f, r2_f, fis_f, used_combinations_f, fit_model_f = fit_data(
                f"{self.structs_in}_out",
                force_np=conc_vals.shape[0],
                explore_all=True,
                p_names_in=self.p_names_in,
                target=self.target,
                regressor=self.regressor,
                silent=self.silent,
                ow_hb_vals=new_hb_vals,
                ow_hy_vals=new_hy_vals,
                ow_sb_vals=new_sb_vals,
                paral=self.paral,
                c_val=self.c_val,
            )
            best_errors.append(mae_f)
            att_imp.append(fis_f[0])
            best_vals.append(conc_vals)
            # sort attributes for their best feature importances
            conc_vals = conc_vals[np.argsort(fis_f[0])[::-1]][:-1]
            new_hb_vals = conc_vals[np.isin(conc_vals, self.hb_vals)]
            new_hy_vals = conc_vals[np.isin(conc_vals, self.hy_vals)]
            new_sb_vals = conc_vals[np.isin(conc_vals, self.sb_vals)]

        # determine the best attribute combination and plot the cource of the search
        performance_order = np.argmin(best_errors)
        best_comb_vals = best_vals[performance_order]
        print(
            f"Best combination of attributes: {' - '.join(best_comb_vals.tolist())}\n"
            f"MAE: {best_errors[performance_order][0]:.4f}"
        )
        if self.plot_search_res:
            # search error plot
            plot_search(best_errors, performance_order, -1)
            # feature importances plot
            fig, ax = plt.subplots(layout="tight")
            x_ = np.arange(len(att_imp[performance_order]))
            att_order = np.argsort(att_imp[performance_order])
            ax.bar(x_, att_imp[performance_order][att_order], color="forestgreen")
            ax.set_xticks(x_, best_comb_vals[att_order], rotation=90)
            plt.show()


if __name__ == "__main__":
    pass
    structs = "af_all"
    pn = np.append(p_names, ["769bc", "N0"])
    temp_cd = np.append(temp_cd, [57.7, 55.6])

    """
    start = timer()
    mae_f, r2_f, fis_f, used_combinations_f, fit_model_f = fit_data(
        f"{structs}_out",
        force_np=2,
        explore_all=True,
        p_names_in=pn,
        target=temp_cd,
        regressor="RI",
        # silent=True,
        ow_hb_vals=new_hb_vals,
        ow_hy_vals=new_hy_vals,
        ow_sb_vals=new_sb_vals,
        paral=-1,
        # save_model="test",
        # c_val=None,
    )
    end_ = timer()
    print(end_ - start)

    """
    AttributeSearch(
        "RF",
        temp_cd,
        pn,
        structs,
        HB_DATA_DESC,
        HY_DATA_DESC,
        SB_DATA_DESC,
        paral=-1,
        c_val=None,
        silent=True,
    ).model_based_search()
    """
    new_hb_vals = HB_DATA_DESC
    new_hy_vals = HY_DATA_DESC
    new_sb_vals = SB_DATA_DESC
    num_par = len(new_hb_vals) + len(new_hy_vals) + len(new_sb_vals)
    regressors = ["KNN", "RI", "LR", "RF"]
    col = ["forestgreen", "royalblue", "orange", "cyan"]
    col_d = dict(zip(regressors, col))

    fig, ax = plt.subplots(figsize=(32, 18))
    start = timer()
    pos = 0
    while num_par > 0:
        for cr, r in enumerate(regressors):
            if num_par == 0:
                break
            mae_f, r2_f, fis_f, used_combinations_f, fit_model_f = fit_data(
                f"{structs}_out",
                force_np=num_par,
                explore_all=True,
                p_names_in=pn,
                target=temp_cd,
                regressor=r,
                silent=True,
                ow_hb_vals=new_hb_vals,
                ow_hy_vals=new_hy_vals,
                ow_sb_vals=new_sb_vals,
                # save_model="test",
                c_val=None,
                paral=-1
            )

            print(f"** {r} {num_par}**\nMAE: {mae_f[0]:.4f}\nR2: {r2_f[0]:.4f}\n")
            if pos == 0:
                lab = r
            else:
                lab = None
            ax.bar(num_par - cr * 0.1, mae_f, width=0.1, label=lab, color=col_d[r])

            if r == "RF":
                # most import features but not the least one
                importance_order = used_combinations_f[0][np.argsort(fis_f[0])[::-1]][
                    :-1
                ]
                new_hb_vals = importance_order[np.isin(importance_order, HB_DATA_DESC)]
                new_hy_vals = importance_order[np.isin(importance_order, HY_DATA_DESC)]
                new_sb_vals = importance_order[np.isin(importance_order, SB_DATA_DESC)]
                num_par = len(new_hb_vals) + len(new_hy_vals) + len(new_sb_vals)
                # print(importance_order)
        pos += 1
    end_ = timer()
    print(end_-start)
    ax.set_xlabel("Number of attributes")
    ax.set_ylabel("Mean absolute error")
    plt.legend()
    # fig.savefig("rf.png")
    plt.show()
    """
