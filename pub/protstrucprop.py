import argparse
from calc_prop import calculate
from regressor_analysis import (
    read_data,
    HB_DATA_DESC,
    HY_DATA_DESC,
    SB_DATA_DESC,
    fit_data,
    AttributeSearch,
)
from predict import predict


def combine_calc(args):
    # protein names and data
    p_names, _ = read_data(args.name_path)
    # calculate properties for all proteins
    calculate(args.out_dir, args.struct_dir, p_names)


def greedy_parameter(args):
    # protein names and data
    p_names, temp_cd = read_data(args.name_path)

    # set the which attributes to test
    new_hb_vals = None
    new_hy_vals = None
    new_sb_vals = None
    if args.explore_all:
        new_hb_vals = HB_DATA_DESC
        new_hy_vals = HY_DATA_DESC
        new_sb_vals = SB_DATA_DESC

    # fit model to data
    mae_f, r2_f, fis_f, used_combinations_f, fit_model_f = fit_data(
        data_path=args.prop_dir,
        target=temp_cd,
        hb_param_num=args.param_number,
        hy_param_num=args.param_number,
        sb_param_num=args.param_number,
        ow_hb_vals=new_hb_vals,
        ow_hy_vals=new_hy_vals,
        ow_sb_vals=new_sb_vals,
        p_names_in=p_names,
        force_np=args.force_nparam,
        explore_all=args.explore_all,
        regressor=args.regressor,
        c_val=args.cross_validation,
        silent=args.silent,
        paral=args.parallel,
    )


def directed_search(args):
    # protein names and data
    p_names, temp_cd = read_data(args.name_path)
    # create search class
    att_search = AttributeSearch(
        regressor=args.regressor,
        target=temp_cd,
        p_names_in=p_names,
        structs_in=args.prop_dir,
        hb_vals=HB_DATA_DESC,
        hy_vals=HY_DATA_DESC,
        sb_vals=SB_DATA_DESC,
        paral=args.parallel,
        c_val=args.cross_validation,
        silent=args.silent,
        plot_search_res=args.search_plot,
    )
    # select search method
    method = args.search_method
    if method == 1:
        att_search.forward_search()
    elif method == 2:
        att_search.backward_search()
    else:
        att_search.model_based_search()


def make_predictions(args):
    p_names, _ = read_data(args.name_path)
    # predict data for proteins in prop_dir
    predict(
        model_name=args.model_name,
        data_dir=args.prop_dir,
        p_names_in=p_names,
        save_results=args.save_results,
        silent=args.silent,
    )


def main():
    # top level parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers()

    # calculate properties parser
    cp_parser = subparsers.add_parser(
        "properties",
        help="calculate properties and create output files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cp_parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="file path to the directory where the output should be stored",
    )
    cp_parser.add_argument(
        "-s",
        "--struct_dir",
        type=str,
        help="file path to the directory where the structures of the proteins "
        "defined in 'name_path' are stored",
    )
    cp_parser.add_argument(
        "-n",
        "--name_path",
        type=str,
        default="protein_data.tsv",
        required=False,
        help="file path where the protein names and data is stored",
    )
    cp_parser.set_defaults(func=combine_calc)

    # greedy parameter search (fit data) parser
    gs_parser = subparsers.add_parser(
        "greedy",
        help="search for the best models over a fixed or ascending number "
        "of parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    gs_parser.add_argument(
        "-n",
        "--name_path",
        type=str,
        default="protein_data.tsv",
        required=False,
        help="file path where the protein names and data is stored",
    )
    gs_parser.add_argument(
        "-p",
        "--prop_dir",
        type=str,
        help="output directory of 'calc_prop' with calculated SaltBridges, "
        "HydrophobicCluster and H-bonds",
    )
    gs_parser.add_argument(
        "-s",
        "--save_model",
        default=None,
        type=str,
        required=False,
        help="How the saved model should be named - None to not save, "
        "directory 'saved_models' will be created if it doesn't exist",
    )
    gs_parser.add_argument(
        "-pn",
        "--param_number",
        type=int,
        default=3,
        required=False,
        help="number of highest correlating H-bonds, salt bridges and "
        "hydrophobic cluster attributes to be tested (per feature)",
    )
    gs_parser.add_argument(
        "-e",
        "--explore_all",
        action="store_true",
        help="set flag to overwrite 'param_num' and use all 27 attributes "
        "to find the best attribute combination",
    )
    gs_parser.add_argument(
        "-fp",
        "--force_nparam",
        type=int,
        required=False,
        default=None,
        help="force number of explored parameters to be of force_np and "
        "not from 1 to sum(*_param_num)",
    )
    gs_parser.add_argument(
        "-r",
        "--regressor",
        type=str,
        required=False,
        default="LR",
        help="- regressor: *  'LR' for linear regression *  'RF' for random forest "
        "*  'KNN' for k-nearest neighbors *  'RI' for Ridge "
        "*  'GB' for GradientBoostingRegressor",
    )
    gs_parser.add_argument(
        "-c",
        "--cross_validation",
        type=int,
        required=False,
        default=None,
        help="integer do specify the number of splits or "
        "None for LeaveOneOut cross validation",
    )
    gs_parser.add_argument(
        "-si",
        "--silent",
        action="store_true",
        help="set flag to suppress output in the terminal",
    )
    gs_parser.add_argument(
        "-pa",
        "--parallel",
        type=int,
        required=False,
        default=None,
        help="None to not parallelize the cross validation, "
        "integer to specify the number of cores or '-1' to use all cores",
    )
    gs_parser.set_defaults(func=greedy_parameter)

    # directed search parser
    ds_parser = subparsers.add_parser(
        "directed",
        help="search for the best model in a directed way",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ds_parser.add_argument(
        "-n",
        "--name_path",
        type=str,
        default="protein_data.tsv",
        required=False,
        help="file path where the protein names and data is stored",
    )
    ds_parser.add_argument(
        "-r",
        "--regressor",
        type=str,
        required=False,
        default="LR",
        help="- regressor: *  'LR' for linear regression *  'RF' for random forest "
        "*  'KNN' for k-nearest neighbors *  'RI' for Ridge "
        "*  'GB' for GradientBoostingRegressor",
    )
    ds_parser.add_argument(
        "-p",
        "--prop_dir",
        type=str,
        help="output directory of 'calc_prop' with calculated SaltBridges, "
        "HydrophobicCluster and H-bonds",
    )
    ds_parser.add_argument(
        "-pa",
        "--parallel",
        type=int,
        required=False,
        default=None,
        help="None to not parallelize the cross validation, "
        "integer to specify the number of cores or '-1' to use all cores",
    )
    ds_parser.add_argument(
        "-c",
        "--cross_validation",
        type=int,
        required=False,
        default=None,
        help="integer do specify the number of splits or "
        "None for LeaveOneOut cross validation",
    )
    ds_parser.add_argument(
        "-si",
        "--silent",
        action="store_true",
        help="set flag to suppress output in the terminal",
    )
    ds_parser.add_argument(
        "-sp",
        "--search_plot",
        action="store_true",
        help="set flag to plot search course and feature importance",
    )
    ds_parser.add_argument(
        "-sm",
        "--search_method",
        type=int,
        default=0,
        required=False,
        help="0 model_based_search, 1 forward_search, 2 backward_search",
    )
    ds_parser.set_defaults(func=directed_search)

    # predict parser
    p_parser = subparsers.add_parser(
        "predict",
        help="predict using a trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_parser.add_argument(
        "-n",
        "--name_path",
        type=str,
        default="protein_data.tsv",
        required=False,
        help="file path where the protein names and data is stored",
    )
    p_parser.add_argument(
        "-m", "--model_name", type=str, help="name of the save model without file type"
    )
    p_parser.add_argument(
        "-p",
        "--prop_dir",
        type=str,
        help="output directory of 'calc_prop' with calculated SaltBridges, "
        "HydrophobicCluster and H-bonds",
    )
    p_parser.add_argument(
        "-sr",
        "--save_results",
        action="store_true",
        help="set flag to store the prediction results in the 'results' directory",
    )
    p_parser.add_argument(
        "-si",
        "--silent",
        action="store_true",
        help="set flag to suppress output in the terminal",
    )
    p_parser.set_defaults(func=make_predictions)

    # parse args and call selected function
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
